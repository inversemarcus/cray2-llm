// Convert one Bonsai Q1_0 GGUF tensor into the Cray-2 matvec tile layout.
// Pure storage/format conversion: no inference arithmetic.
//
// Tile layout (words, host-order uint64 holding the 64-bit Cray word):
//   for each 64-row tile t:
//     for each group pair gp (groups 2gp, 2gp+1; 128 columns each):
//       S[64]   : hi32 = Cray-float(scale[row][2gp]) top half, lo32 = same for 2gp+1
//       W[2gp][0][64], W[2gp][1][64], W[2gp+1][0][64], W[2gp+1][1][64]
//   Each W word holds 64 weight signs for one row; element j of the 64-column
//   chunk is bit (63-j).  1 = +scale, 0 = -scale (ggml dequantize_row_q1_0).
// Rows beyond m are padded with zero bits and zero scale.
//
// Embedding layout ("rows" mode): row-major, per row per group: S, W0, W1 where
// S is a full Cray float scale word.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static uint64_t half_to_cray(uint16_t h) {
  uint64_t sign = (uint64_t)(h >> 15) << 63;
  int e = (h >> 10) & 31;
  uint32_t f = h & 1023;
  if (e == 31) { fprintf(stderr, "nonfinite scale\n"); exit(1); }
  if (!e) {
    if (!f) return 0;
    e = -13;
    while (!(f & 1024)) { f <<= 1; e--; }
    return sign | ((uint64_t)(e + 16384) << 48) | ((uint64_t)f << 37);
  }
  return sign | ((uint64_t)(e - 14 + 16384) << 48) | ((uint64_t)(f | 1024) << 37);
}

// 8 little-endian bytes of Q1 bits (element j = byte j/8 bit j%8) -> word with
// element j at bit 63-j.
static uint64_t bits_msb_first(const uint8_t *p) {
  uint64_t w = 0;
  for (int j = 0; j < 64; j++)
    if ((p[j >> 3] >> (j & 7)) & 1) w |= 1ULL << (63 - j);
  return w;
}

int main(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr, "usage: convert_q1 gguf abs_offset rows cols image word_base tiles|rows\n");
    return 2;
  }
  int fd = open(argv[1], O_RDONLY);
  if (fd < 0) { perror("open gguf"); return 1; }
  struct stat st; fstat(fd, &st);
  const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  size_t off = strtoull(argv[2], 0, 10), m = strtoull(argv[3], 0, 10), n = strtoull(argv[4], 0, 10);
  size_t base = strtoull(argv[6], 0, 10);
  int tiles = !strcmp(argv[7], "tiles");
  size_t G = n / 128;
  if (n % 256 || off + m * G * 18 > (size_t)st.st_size) { fprintf(stderr, "shape/bounds\n"); return 1; }
  int out = open(argv[5], O_RDWR);
  if (out < 0) { perror("open image"); return 1; }
  if (tiles) {
    size_t T = (m + 63) / 64, tilewords = G * 160;
    uint64_t *buf = malloc(tilewords * 8);
    for (size_t t = 0; t < T; t++) {
      for (size_t gp = 0; gp < G / 2; gp++) {
        uint64_t *blk = buf + gp * 320;
        for (size_t r = 0; r < 64; r++) {
          size_t row = t * 64 + r;
          uint64_t s0 = 0, s1 = 0, w[4] = {0, 0, 0, 0};
          if (row < m) {
            const uint8_t *b0 = data + off + (row * G + 2 * gp) * 18, *b1 = b0 + 18;
            s0 = half_to_cray(b0[0] | (uint16_t)b0[1] << 8);
            s1 = half_to_cray(b1[0] | (uint16_t)b1[1] << 8);
            if ((s0 & 0xffffffffULL) || (s1 & 0xffffffffULL)) { fprintf(stderr, "scale needs >32 bits\n"); return 1; }
            w[0] = bits_msb_first(b0 + 2); w[1] = bits_msb_first(b0 + 10);
            w[2] = bits_msb_first(b1 + 2); w[3] = bits_msb_first(b1 + 10);
          }
          blk[r] = s0 | (s1 >> 32);
          for (int k = 0; k < 4; k++) blk[64 + 64 * k + r] = w[k];
        }
      }
      if (pwrite(out, buf, tilewords * 8, (off_t)(base + t * tilewords) * 8) != (ssize_t)(tilewords * 8)) { perror("pwrite"); return 1; }
    }
    printf("{\"rows\":%zu,\"cols\":%zu,\"tiles\":%zu,\"words\":%zu}\n", m, n, T, T * tilewords);
  } else {
    size_t rowwords = G * 3;
    uint64_t *buf = malloc(rowwords * 8);
    for (size_t row = 0; row < m; row++) {
      for (size_t g = 0; g < G; g++) {
        const uint8_t *b = data + off + (row * G + g) * 18;
        buf[3 * g] = half_to_cray(b[0] | (uint16_t)b[1] << 8);
        buf[3 * g + 1] = bits_msb_first(b + 2);
        buf[3 * g + 2] = bits_msb_first(b + 10);
      }
      if (pwrite(out, buf, rowwords * 8, (off_t)(base + row * rowwords) * 8) != (ssize_t)(rowwords * 8)) { perror("pwrite"); return 1; }
    }
    printf("{\"rows\":%zu,\"cols\":%zu,\"words\":%zu}\n", m, n, m * rowwords);
  }
  return 0;
}
