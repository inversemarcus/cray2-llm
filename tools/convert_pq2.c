// Convert one Ternary-Bonsai PQ2_0 GGUF tensor into the Cray-2 ternary layout.
// Storage/format conversion only: no inference arithmetic.
//
// PQ2_0 block (128 weights): fp16 d, 32 bytes of 2-bit codes; element j is
// (qs[j/4] >> 2*(j%4)) & 3 with 0 = -1, 1 = 0, 2 = +1 (3 never occurs).
//
// Tile layout (host-order uint64 words), for each 64-row tile, for each quad
// of 4 consecutive 128-column groups:
//   (all of a tile's scale vectors first, then all of its mask groups)
//   S[64]  : four 16-bit scale fields per row; group q of the quad in bits
//            (63-16q)..(48-16q).  Field = (E5<<11)|(1<<10)|m10 with
//            E5 = 18+e (fp16 exponent e, must be 1..13), so that
//            ((field) << 37) | (511<<53) is the exact Cray float of d.
//            A zero scale is field 0.
//   then per group: POS0[64] POS1[64] NNEG0[64] NNEG1[64]
//            POS = (w == +1), NNEG = (w != -1); column halves 0..63, 64..127;
//            element j of a half is bit (63-j).
// Rows mode (embedding, disk image): per row per group: Cray float d, POS0,
//            POS1, NNEG0, NNEG1.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static uint16_t field_of(uint16_t h) {
  if (!(h & 0x7fff)) return 0;
  int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  if (e == 0) {            // fp16 subnormal: renormalize exactly (<= 10 significant bits)
    e = 1;
    while (!(m & 1024)) { m <<= 1; e--; }
    m &= 1023;
  }
  if (s || e > 13 || 18 + e < 1) { fprintf(stderr, "unsupported scale %04x\n", h); exit(1); }
  return (uint16_t)(((18 + e) << 11) | (1 << 10) | m);
}

static uint64_t half_to_cray(uint16_t h) {
  uint16_t f = field_of(h);
  return f ? (((uint64_t)f << 37) | (511ULL << 53)) : 0;
}

static void masks(const uint8_t *qs, int half, uint64_t *pos, uint64_t *nneg) {
  uint64_t p = 0, n = 0;
  for (int j = 0; j < 64; j++) {
    int e = half * 64 + j;
    int c = (qs[e >> 2] >> (2 * (e & 3))) & 3;
    if (c == 3) { fprintf(stderr, "code 3\n"); exit(1); }
    if (c == 2) p |= 1ULL << (63 - j);
    if (c != 0) n |= 1ULL << (63 - j);
  }
  *pos = p; *nneg = n;
}

int main(int argc, char **argv) {
  if (argc != 8) { fprintf(stderr, "usage: convert_pq2 gguf off rows cols image word_base tiles|rows\n"); return 2; }
  int fd = open(argv[1], O_RDONLY);
  struct stat st; fstat(fd, &st);
  const uint8_t *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  size_t off = strtoull(argv[2], 0, 10), m = strtoull(argv[3], 0, 10), n = strtoull(argv[4], 0, 10);
  size_t base = strtoull(argv[6], 0, 10);
  int tiles = !strcmp(argv[7], "tiles");
  size_t G = n / 128;
  const size_t BB = 34;  // block bytes
  if (n % 512 || off + m * G * BB > (size_t)st.st_size) { fprintf(stderr, "shape/bounds\n"); return 1; }
  int out = open(argv[5], O_RDWR);
  if (tiles) {
    size_t T = (m + 63) / 64, tw = G / 4 * (64 + 4 * 256);
    uint64_t *buf = malloc(tw * 8);
    for (size_t t = 0; t < T; t++) {
      for (size_t qd = 0; qd < G / 4; qd++) {
        uint64_t *svec = buf + qd * 64, *mk = buf + 16 * G + qd * 1024;
        for (size_t r = 0; r < 64; r++) {
          size_t row = t * 64 + r;
          uint64_t sw = 0;
          for (int q = 0; q < 4; q++) {
            uint64_t p0 = 0, p1 = 0, n0 = 0, n1 = 0;
            if (row < m) {
              const uint8_t *b = data + off + (row * G + qd * 4 + q) * BB;
              sw |= (uint64_t)field_of(b[0] | (uint16_t)b[1] << 8) << (48 - 16 * q);
              masks(b + 2, 0, &p0, &n0);
              masks(b + 2, 1, &p1, &n1);
            }
            uint64_t *g = mk + q * 256;
            g[r] = p0; g[64 + r] = p1; g[128 + r] = n0; g[192 + r] = n1;
          }
          svec[r] = sw;
        }
      }
      if (pwrite(out, buf, tw * 8, (off_t)(base + t * tw) * 8) != (ssize_t)(tw * 8)) { perror("pwrite"); return 1; }
    }
    printf("{\"rows\":%zu,\"cols\":%zu,\"tiles\":%zu,\"words\":%zu}\n", m, n, T, T * tw);
  } else {
    size_t rw = G * 5;
    uint64_t *buf = malloc(rw * 8);
    for (size_t row = 0; row < m; row++) {
      for (size_t g = 0; g < G; g++) {
        const uint8_t *b = data + off + (row * G + g) * BB;
        buf[5 * g] = half_to_cray(b[0] | (uint16_t)b[1] << 8);
        masks(b + 2, 0, &buf[5 * g + 1], &buf[5 * g + 3]);
        masks(b + 2, 1, &buf[5 * g + 2], &buf[5 * g + 4]);
      }
      if (pwrite(out, buf, rw * 8, (off_t)(base + row * rw) * 8) != (ssize_t)(rw * 8)) { perror("pwrite"); return 1; }
    }
    printf("{\"rows\":%zu,\"cols\":%zu,\"words\":%zu}\n", m, n, m * rw);
  }
  return 0;
}
