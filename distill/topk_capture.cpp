// Record a teacher's top-k next-token log-probabilities over training chunks,
// using llama.cpp (CPU or GPU).  Same output as distill/teacher_topk.py, as
// two .npy files per shard: ids (uint32 [n,512,k]) and logp (float16 [n,512,k]).
//
//   topk_capture MODEL.gguf CHUNKS.npy OUT_PREFIX [k=32] [threads]
//
// CHUNKS.npy is uint32 [n, 513] from distill/prepare_wiki.py; the first 512
// tokens of each row are read with no BOS (as the MLX path does).  Rows already
// present in OUT_PREFIX.partial are skipped, so an interrupted run resumes.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> load_npy_u32(const char *path, size_t &rows, size_t &cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open chunks");
    char magic[8]; f.read(magic, 8);
    uint16_t hl; f.read((char *)&hl, 2);
    std::string h(hl, ' '); f.read(&h[0], hl);
    if (h.find("<u4") == std::string::npos || h.find("False") == std::string::npos) throw std::runtime_error("need C-order uint32");
    size_t a = h.find('(') + 1, b = h.find(',', a), c = h.find(')', b);
    rows = std::stoul(h.substr(a, b - a)); cols = std::stoul(h.substr(b + 1, c - b - 1));
    std::vector<uint32_t> v(rows * cols);
    f.read((char *)v.data(), v.size() * 4);
    return v;
}

static void write_npy(const std::string &path, const char *descr, const std::vector<size_t> &shape, const void *data, size_t bytes) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); i++) s += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? ", " : "");
    s += ")";
    std::string h = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + s + ", }";
    while ((10 + h.size() + 1) % 64) h += ' ';
    h += '\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    uint16_t hl = (uint16_t)h.size(); f.write((char *)&hl, 2);
    f.write(h.data(), h.size());
    f.write((const char *)data, bytes);
}

static uint16_t to_f16(float x) {  // IEEE half, round to nearest even
    uint32_t u; std::memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000; int e = (int)((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffff;
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) {
        if (e < -10) return sign;
        m |= 0x800000; int sh = 14 - e;
        uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return sign | r;
    }
    uint32_t r = m >> 13, rem = m & 0x1fff;
    uint32_t out = sign | (e << 10) | r;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) out++;
    return out;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: topk_capture MODEL CHUNKS.npy OUT_PREFIX [k] [threads]\n"); return 2; }
    const int K = argc > 4 ? atoi(argv[4]) : 32, T = argc > 5 ? atoi(argv[5]) : 12;
    size_t rows, cols;
    auto chunks = load_npy_u32(argv[2], rows, cols);
    const int N = (int)cols - 1;
    std::string out = argv[3];
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto mp = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 3; }
    auto cp = llama_context_default_params();
    cp.n_ctx = N; cp.n_batch = N; cp.n_ubatch = N; cp.n_threads = T; cp.n_threads_batch = T;
    llama_context *ctx = llama_init_from_model(model, cp);
    const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<uint32_t> ids(rows * N * K);
    std::vector<uint16_t> lp(rows * N * K);
    // resume
    size_t start = 0;
    {
        std::ifstream pf(out + ".partial", std::ios::binary);
        if (pf) {
            pf.read((char *)&start, sizeof start);
            pf.read((char *)ids.data(), start * N * K * 4);
            pf.read((char *)lp.data(), start * N * K * 2);
            if (!pf) start = 0;
        }
    }
    llama_batch batch = llama_batch_init(N, 0, 1);
    std::vector<int> idx(V);
    for (size_t r = start; r < rows; r++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = N;
        for (int i = 0; i < N; i++) {
            batch.token[i] = (llama_token)chunks[r * cols + i];
            batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed at row %zu\n", r); return 4; }
        for (int i = 0; i < N; i++) {
            const float *lg = llama_get_logits_ith(ctx, i);
            float mx = *std::max_element(lg, lg + V);
            double se = 0; for (int v = 0; v < V; v++) se += std::exp((double)lg[v] - mx);
            float lse = mx + (float)std::log(se);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            for (int k = 0; k < K; k++) {
                ids[(r * N + i) * K + k] = (uint32_t)idx[k];
                lp[(r * N + i) * K + k] = to_f16(lg[idx[k]] - lse);
            }
        }
        if ((r + 1) % 16 == 0 || r + 1 == rows) {
            std::ofstream pf(out + ".partial.tmp", std::ios::binary);
            size_t done = r + 1;
            pf.write((char *)&done, sizeof done);
            pf.write((char *)ids.data(), done * N * K * 4);
            pf.write((char *)lp.data(), done * N * K * 2);
            pf.close();
            std::rename((out + ".partial.tmp").c_str(), (out + ".partial").c_str());
            fprintf(stderr, "%s: %zu/%zu rows\n", argv[2], done, rows);
        }
    }
    write_npy(out + ".ids.npy", "<u4", {rows, (size_t)N, (size_t)K}, ids.data(), ids.size() * 4);
    write_npy(out + ".logp.npy", "<f2", {rows, (size_t)N, (size_t)K}, lp.data(), lp.size() * 2);
    std::remove((out + ".partial").c_str());
    llama_batch_free(batch);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
