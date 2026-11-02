// Native llama.cpp reference only. This executable is not Cray inference.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
static std::string quote(const std::string &s) { std::string r="\""; for(char c:s){if(c=='\\'||c=='\"')r+='\\'; if(c=='\n')r+="\\n";else r+=c;} return r+'\"'; }
struct Capture {
 fs::path path; std::set<const ggml_tensor*> visited; std::set<std::string> saved,observed; std::vector<std::string> rows; std::string error;
 bool want(const std::string &n) const {
  static const std::set<std::string> exact={"ffn_up-0","ffn_gate-0","ffn_down-0","attn_out-0","attn_norm-0","Qcur_normed-0","Kcur_normed-0","Qcur-0","Kcur-0","Vcur-0","ffn_inp-0","ffn_norm-0","ffn_out-0","result_norm","result_output","inp_embd"};
  return exact.count(n)||n.rfind("l_out-",0)==0;
 }
 void write(const std::string &name,const std::vector<float>&v,const int64_t *ne,const size_t *nb,const std::string&type,const std::string&source="") {
  if(!saved.insert(name).second)return;
  std::string filename=name+".f32.bin";std::replace(filename.begin(),filename.end(),'/','_');
  std::ofstream file(path/filename,std::ios::binary);file.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));if(!file)throw std::runtime_error("write failed "+filename);
  float lo=*std::min_element(v.begin(),v.end()),hi=*std::max_element(v.begin(),v.end());size_t finite=0;for(float x:v)finite+=std::isfinite(x);
  std::ostringstream s;s<<std::setprecision(9)<<"{\"name\":"<<quote(name)<<",\"file\":"<<quote(filename)<<",\"source_name\":"<<quote(source.empty()?name:source)<<",\"dtype\":\"float32_le\",\"original_type\":"<<quote(type)<<",\"elements\":"<<v.size()<<",\"shape\":[";
  for(int i=0;i<4;i++)s<<(i?",":"")<<ne[i];s<<"],\"original_byte_strides\":[";for(int i=0;i<4;i++)s<<(i?",":"")<<nb[i];
  s<<"],\"layout\":\"dense_dim0_fastest\",\"min\":"<<lo<<",\"max\":"<<hi<<",\"finite_elements\":"<<finite<<"}";rows.push_back(s.str());
 }
 void scan(ggml_tensor*t) {
  if(!t||!visited.insert(t).second)return;
  observed.insert(t->name);
  if(std::string(t->name)=="Qcur-0"||std::string(t->name)=="Kcur-0")std::cerr<<"GOLDEN_NODE "<<t->name<<" "<<ggml_op_desc(t)<<" "<<t->ne[0]<<","<<t->ne[1]<<","<<t->ne[2]<<","<<t->ne[3]<<"\n";
  if(t->op==GGML_OP_ROPE&&(std::string(t->name)=="Qcur-0"||std::string(t->name)=="Kcur-0"||std::string(t->name)=="k_cache_in-0")) {
   std::ofstream params(path/(std::string(t->name)+".ropeparams.bin"),std::ios::binary);
   params.write(reinterpret_cast<const char*>(t->op_params),sizeof(t->op_params));
  }
  std::string capture_name=t->name;
  if((capture_name=="Qcur-0"||capture_name=="Kcur-0")&&t->op!=GGML_OP_ROPE)capture_name=capture_name.substr(0,4)+"_projection-0";
  if(std::string(t->name)=="k_cache_in-0"&&t->op==GGML_OP_ROPE)capture_name="Kcur-0";
  if((want(t->name)||std::string(t->name)=="k_cache_in-0")&&!saved.count(capture_name)) {
   if(!t->data)throw std::runtime_error(std::string("no computed data: ")+t->name);
   if(t->type!=GGML_TYPE_F32&&t->type!=GGML_TYPE_F16&&t->type!=GGML_TYPE_BF16)throw std::runtime_error(std::string("unsupported capture type: ")+t->name);
   std::vector<unsigned char> data(ggml_nbytes(t));
   if(t->buffer&&ggml_backend_buffer_is_host(t->buffer))std::memcpy(data.data(),t->data,data.size());else ggml_backend_tensor_get(t,data.data(),0,data.size());
   std::vector<float> v;v.reserve(ggml_nelements(t));
   for(int64_t i3=0;i3<t->ne[3];i3++)for(int64_t i2=0;i2<t->ne[2];i2++)for(int64_t i1=0;i1<t->ne[1];i1++)for(int64_t i0=0;i0<t->ne[0];i0++) {
    const auto *p=data.data()+i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3];float value;
    if(t->type==GGML_TYPE_F32)std::memcpy(&value,p,4);else if(t->type==GGML_TYPE_F16){ggml_fp16_t x;std::memcpy(&x,p,2);value=ggml_fp16_to_fp32(x);}else{ggml_bf16_t x;std::memcpy(&x,p,2);value=ggml_bf16_to_fp32(x);}v.push_back(value);
   }
   write(capture_name,v,t->ne,t->nb,ggml_type_name(t->type),t->name);
  }
  // View tensors may not receive scheduler callbacks. Capture them at the
  // earliest completed consumer; all ancestor values have been computed.
  for(auto*src:t->src)scan(src);
 }
 static bool callback(ggml_tensor*t,bool ask,void*arg) {
  if(ask)return true; // one callback boundary per op prevents missed sources
  auto*c=static_cast<Capture*>(arg);try{c->scan(t);return true;}catch(const std::exception&e){c->error=e.what();return false;}
 }
};
int main(int argc,char**argv) {
 try {
  if(argc<3){std::cerr<<"usage: capture MODEL.gguf OUTPUT_DIRECTORY [TOKEN_ID=198]\n";return 2;}
  uint16_t endian=1;if(*reinterpret_cast<unsigned char*>(&endian)!=1)throw std::runtime_error("little-endian host required");
  llama_token token=argc>3?std::stoi(argv[3]):198;Capture cap;cap.path=argv[2];fs::create_directories(cap.path);
  llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
  llama_model*model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("model load failed");
  int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));if(token<0||token>=nv)throw std::runtime_error("token out of range");
  auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=1;cp.n_ubatch=1;cp.n_threads=1;cp.n_threads_batch=1;cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=false;cp.op_offload=false;cp.cb_eval=Capture::callback;cp.cb_eval_user_data=&cap;
  auto*ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context init failed");
  int rc=llama_decode(ctx,llama_batch_get_one(&token,1));if(rc||!cap.error.empty())throw std::runtime_error("decode failed "+std::to_string(rc)+" "+cap.error);
  const float*lp=llama_get_logits_ith(ctx,-1);if(!lp)throw std::runtime_error("missing final logits");
  std::vector<float>logits(lp,lp+nv);int64_t ne[4]={nv,1,1,1};size_t nb[4]={4,size_t(nv)*4,size_t(nv)*4,size_t(nv)*4};cap.write("final_logits",logits,ne,nb,"f32");
  const std::vector<std::string>required={"attn_norm-0","Qcur_normed-0","Kcur_normed-0","Qcur-0","Kcur-0","Vcur-0","ffn_inp-0","ffn_norm-0","ffn_out-0","l_out-0","result_norm","final_logits"};std::vector<std::string>missing;for(auto&n:required)if(!cap.saved.count(n))missing.push_back(n);
  std::ofstream obs(cap.path/"observed-names.txt");for(auto&n:cap.observed)obs<<n<<'\n';
  std::ofstream manifest(cap.path/"manifest.json");manifest<<"{\n\"reference_only\":true,\"engine_revision\":\"a15369706017f177cae8aa40b6b8592259a48806\",\"model\":"<<quote(fs::absolute(argv[1]).string())<<",\"token_id\":"<<token<<",\"position\":0,\"context\":2048,\"threads\":1,\"kv_type\":\"f32\",\"flash_attention\":false,\"vocab_size\":"<<nv<<",\"argmax_token\":"<<std::distance(logits.begin(),std::max_element(logits.begin(),logits.end()))<<",\"missing_required\":[";
  for(size_t i=0;i<missing.size();i++)manifest<<(i?",":"")<<quote(missing[i]);manifest<<"],\n\"tensors\":[\n";for(size_t i=0;i<cap.rows.size();i++)manifest<<(i?",\n":"")<<cap.rows[i];manifest<<"\n]}\n";
  llama_free(ctx);llama_model_free(model);llama_backend_free();std::cout<<"captured "<<cap.rows.size()<<" tensors; missing "<<missing.size()<<"; manifest "<<cap.path/"manifest.json"<<'\n';return missing.empty()?0:3;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
