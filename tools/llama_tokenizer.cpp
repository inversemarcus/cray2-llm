// Vocabulary-only helper. No context, inference, tensor compute or llama_decode.
#include "llama.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void read(void*p,size_t n){std::cin.read((char*)p,n);if(!std::cin)throw std::runtime_error("short input");}
static void reply(char status,const void*p,size_t n){std::cout.put(status);uint64_t size=n;std::cout.write((char*)&size,8);if(n)std::cout.write((const char*)p,n);std::cout.flush();}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
 llama_log_set([](ggml_log_level,const char*,void*){},nullptr);
 auto params=llama_model_default_params();params.vocab_only=true;
 auto model=llama_model_load_from_file(argv[1],params);if(!model)return 3;
 auto vocab=llama_model_get_vocab(model);
 try{
  for(;;){char op;if(!std::cin.get(op))break;uint64_t n;read(&n,8);
   if(op=='Q'){reply('O',nullptr,0);break;}
   if(op=='E'){
    std::string text(n,'\0');read(&text[0],n);
    int size=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
    std::vector<llama_token> tokens(size<0?-size:size);
    size=llama_tokenize(vocab,text.data(),text.size(),tokens.data(),tokens.size(),false,true);
    if(size<0)throw std::runtime_error("tokenize");reply('O',tokens.data(),size*4);
   }else if(op=='D'){
    std::vector<llama_token> tokens(n);read(tokens.data(),n*4);
    int size=llama_detokenize(vocab,tokens.data(),n,nullptr,0,false,false);
    std::vector<char> text(size<0?-size:size);
    size=llama_detokenize(vocab,tokens.data(),n,text.data(),text.size(),false,false);
    if(size<0)throw std::runtime_error("detokenize");reply('O',text.data(),size);
   }else if(op=='G'){if(n>=uint64_t(llama_vocab_n_tokens(vocab)))throw std::runtime_error("token index");uint8_t yes=llama_vocab_is_eog(vocab,llama_token(n));reply('O',&yes,1);}
   else throw std::runtime_error("command");
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';llama_model_free(model);return 1;}
 llama_model_free(model);
}
