"""Generate the CRAY-2 CAL program for Ternary-Bonsai-8B (PQ2_0 weights, native
Q8_K activation semantics: 8-bit, blocks of 256, F32 rounding reproduced).

All neural arithmetic is performed by the generated Cray-2 instructions.
The host (Foreground Processor) only supplies token IDs/positions, the
weight image and constant tables (RoPE cos/sin), and reads results.

Register conventions: A0 = 0 and A2 = 1 always; S0 = 0 always.
Calls:  A6 <target> ; J A6      Return: J A6
"""
import json, math, pathlib, struct, sys

from .calgen import Gen, cf, rope_table

MW = 1 << 20

# ---------------------------------------------------------------- geometry
L, HID, FF, NH, NKV, HD = 36, 4096, 12288, 32, 8, 128
VOCAB = 151669
HEAD_TILES = (VOCAB + 63) // 64          # 2370
NCPU = 4
CTX = 200                                # max positions (KV cache; memory-limited)
CTXS = CTX + 1                           # K^T row stride (odd: spreads banks)
EPS = 9.999999974752427e-07

# ---------------------------------------------------------------- CM map
MBOX = 0x100000      # [0] token [1] pos [2] out token [3] debug flag [8] barrier count
BARC = MBOX + 8
PART = MBOX + 32     # argmax partials: 2 words per CPU
X = 0x101000         # residual stream (4096)
XN = 0x102000        # normalized activations (4096)
QKV = 0x103000       # q 4096 | k 1024 | v 1024
ATT = 0x105000       # attention output (4096)
H = 0x106000         # SwiGLU output (12288)
QTA = 0x10A000       # plane tables (96 groups x 64 words)
QTB = 0x10C000
EMB = 0x10E000       # embedding row read from disk by the Foreground Processor (160 words)
LOGITS = 0x110000    # 151680
ROPE = 0x140000      # pos*128: cos[64] sin[64] (x mscale)
DESC = 0x150000      # per-layer descriptors (16 words)
QTW = 64             # words per 128-group plane-table entry
DESC_W = 16

# ---------------------------------------------------------------- LM map
LM = dict(cpu=0, bseq=1, link0=2, link1=3, link2=4, link3=5,
          tok=10, pos=11, layer=12, desc=13, tcount=14, outp=15, tile=16,
          mvqt=17, mvgp=18, wptr=19, nscal=20, nvec=21, src=22, wn=23,
          grp=24, qtp=25, kh=26, h=27, sc=28, n=29, tmp0=30, tmp1=31,
          tags=32,   # 6 words: tag for planes 0..5
          mhi=40, absm=41, one=42, two=43, c31=44, r31=45, hdsc=46, eps=47,
          log2e=48, ln2=49, ninf=50, coefm=51, half=52, invh=53, invf=54,
          ectx=55, kvoff=56, gptr=57, uptr=58, hptr=59, ktb=60, vtb=61, rope=62, sval=63,
          lmt=64, lmt2=128, qh=256, scores=1024, am=6144, ai=6208, gt=6272, iota=6336,
          expc=6400, c23=6420, m24=6421, b47=6422, pcnt=6423, yacc=6464)
LM = {k: v + 64 for k, v in LM.items()}   # LM 0..63 holds the matvec Y accumulator (addressed by A0)
LM.update(yacc=0, xq=8192, smask=8460, scon=8461, mvsw=8462, iscl=8463, tacc=8464)
NPL = 8                                    # activation bit planes (8-bit two's complement)
QMAX = 127                                 # native Q8_K: iscale = -127/max

EXPDEG = 11
SLOT0, SLOT1, DTAB = 9216, 9472, 10240      # LM: mask double buffer, decoded scale table


def build(layout, act_bits=6):
    """act_bits = 8: native Q8_K (iscale = -127/max); 6: same quantizer with
    iscale = -31/max and codes in [-31, 31]."""
    global NPL, QMAX
    NPL, QMAX = {8: (8, 127), 6: (6, 31)}[act_bits]
    g = Gen(LM, MBOX)
    e = g.e
    T = layout['tensors']
    KVBASE = (layout['end'] + 1023) // 1024 * 1024
    KT_L = NKV * HD * CTXS            # K^T words per layer
    V_L = CTX * NKV * HD              # V words per layer
    KV_L = KT_L + V_L
    assert KVBASE + L * KV_L < 256 * MW, 'KV cache does not fit'
    tile_w = lambda cols: cols // 128 * 272

    # ================================================================ start
    g.lab('start')
    e('A0 0')
    e('A2 1')
    e('S0 0')
    e('S1 A7')
    e('[%d] S1' % LM['cpu'])
    e('[%d] S0' % LM['bseq'])
    for p in range(NPL):
        tag = ((0o40060 + p) << 48) | ((1 << 63) if p == NPL - 1 else 0)
        e('S1 %d' % tag)
        e('[%d] S1' % (LM['tags'] + p))
    consts = dict(mhi=0xFFFFFFFF00000000, absm=0x7FFFFFFFFFFFFFFF, one=cf(1.0), two=cf(2.0),
                  c31=cf(31.0), r31=cf(1.0 / 31.0), hdsc=cf(1.0 / math.sqrt(HD)), eps=cf(EPS),
                  log2e=cf(1.0 / math.log(2.0)), ln2=cf(math.log(2.0)), ninf=cf(-1e300),
                  coefm=0x0000FFFFFFFFFFFF, half=cf(0.5), invh=cf(1.0 / HID), invf=cf(1.0 / FF),
                  c23=(1 << 23) - 1, m24=0xFFFFFFFFFF000000, b47=1 << 47,
                  smask=((1 << 16) - 1) << 37, scon=511 << 53)
    for k, v in consts.items():
        e('S1 %d' % v)
        e('[%d] S1' % LM[k])
    for i in range(EXPDEG + 1):  # exp Taylor coefficients 1/i!, highest first
        e('S1 %d' % cf(1.0 / math.factorial(EXPDEG - i)))
        e('[%d] S1' % (LM['expc'] + i))
    # iota vector 0..63 (integers) in LM
    e('S1 -1')
    e('S2 1')
    e('V1 CI,S1&S2')
    e('A3 %d' % LM['iota'])
    e('[A3] V1')

    # ================================================================ token loop
    g.lab('tokloop')
    e('A1 [%d]' % LM['cpu'])
    e('JN A1,tok_wait')
    e('EXIT 1')                      # Foreground: supply token and position
    g.lab('tok_wait')
    g.call('barrier')
    e('S1 (%d)' % MBOX)
    e('[%d] S1' % LM['tok'])
    e('S1 (%d)' % (MBOX + 1))
    e('[%d] S1' % LM['pos'])
    # ---- embedding: the Foreground Processor has read the token's row from
    # disk into EMB (per group: d, POS0, POS1, NNEG0, NNEG1).  Own 8 groups.
    e('A4 [%d]' % LM['cpu'])
    e('A3 40')
    e('A1 A4*A3')
    e('A3 %d' % EMB)
    e('A1 A1+A3')
    e('A3 1024')
    e('A3 A4*A3')
    e('A4 %d' % X)
    e('A4 A4+A3')                    # X + 1024*cpu
    e('A3 64')
    e('VL A3')
    e('V7 V6\\V6')                   # zero vector
    e('A3 8')
    g.lab('emb_loop')
    e('S1 (A1)')                     # d
    e('S2 S0-FS1')
    e('V1 S2+FV7')                   # broadcast -d
    for h in (0, 1):
        e('S3 (A1,%d)' % (3 + h))    # NNEG half h
        e('VM S3')
        e('V2 S0!V1&VM')             # not-negative -> 0, else -d
        e('S3 (A1,%d)' % (1 + h))    # POS half h
        e('VM S3')
        e('V3 S1!V2&VM')             # positive -> +d
        e('(A4,A2) V3')
        e('A5 64')
        e('A4 A4+A5')
    e('A5 5')
    e('A1 A1+A5')
    e('A3 A3-A2')
    e('JN A3,emb_loop')
    g.call('barrier')
    g.dbg(0)

    # ================================================================ layers
    e('A1 %d' % DESC)
    e('[%d] A1' % LM['desc'])
    e('A1 %d' % L)
    e('[%d] A1' % LM['layer'])
    g.lab('layer_loop')
    # ---- attention norm + quantize (own 8 groups) -> QTA
    e('A4 [%d]' % LM['desc'])
    e('S1 (A4,5)')
    e('[%d] S1' % LM['wn'])
    e('A1 %d' % X)
    e('A3 %d' % QTA)
    g.call('normq4096')
    g.call('barrier')
    g.dbg(1)
    # ---- QKV projection: 96 tiles, 24 per CPU -> QKV
    e('A4 [%d]' % LM['desc'])
    e('S1 (A4,0)')
    e('A1 S1')
    e('A3 %d' % QKV)
    e('A4 24')
    e('A5 %d' % QTA)
    e('A7 32')
    g.call('mv_range_store')
    g.call('barrier')
    g.dbg(2)
    # ---- attention (own 2 kv heads, 8 q heads) -> ATT, QTB
    g.call('attention')
    g.call('barrier')
    g.dbg(3)
    # ---- output projection: 64 tiles, 16 per CPU, X += .
    e('A4 [%d]' % LM['desc'])
    e('S1 (A4,1)')
    e('A1 S1')
    e('A3 %d' % X)
    e('A4 16')
    e('A5 %d' % QTB)
    e('A7 32')
    g.call('mv_range_add')
    g.call('barrier')
    g.dbg(4)
    # ---- ffn norm + quantize -> QTA
    e('A4 [%d]' % LM['desc'])
    e('S1 (A4,6)')
    e('[%d] S1' % LM['wn'])
    e('A1 %d' % X)
    e('A3 %d' % QTA)
    g.call('normq4096')
    g.call('barrier')
    g.dbg(5)
    # ---- gate/up + SwiGLU (48 tiles per CPU) -> H, quantize own 24 groups -> QTB
    g.call('ffn_gateup')
    g.call('barrier')
    g.dbg(6)
    # ---- down projection: 64 tiles, 16 per CPU, 48 pairs, X += .
    e('A4 [%d]' % LM['desc'])
    e('S1 (A4,4)')
    e('A1 S1')
    e('A3 %d' % X)
    e('A4 16')
    e('A5 %d' % QTB)
    e('A7 96')
    g.call('mv_range_add')
    g.call('barrier')
    g.dbg(7)
    # ---- optional debug stop after each layer (CPU0, debug flag set)
    e('A1 [%d]' % LM['cpu'])
    e('JN A1,no_dbg')
    e('S1 (%d)' % (MBOX + 3))
    e('JZ S1,no_dbg')
    e('A1 [%d]' % LM['layer'])
    e('EXIT 20')
    g.lab('no_dbg')
    e('A4 [%d]' % LM['desc'])
    e('A3 %d' % DESC_W)
    e('A4 A4+A3')
    e('[%d] A4' % LM['desc'])
    e('A1 [%d]' % LM['layer'])
    e('A1 A1-A2')
    e('[%d] A1' % LM['layer'])
    e('JN A1,layer_loop')

    # ================================================================ head
    e('S1 %d' % T['output_norm.weight']['addr'])
    e('[%d] S1' % LM['wn'])
    e('A1 %d' % X)
    e('A3 %d' % QTA)
    g.call('normq4096')
    g.call('barrier')
    g.call('head')
    g.call('barrier')
    e('A1 [%d]' % LM['cpu'])
    e('JN A1,tokloop')
    # CPU0 combines partials (first strictly greater wins: lower CPU = lower rows)
    e('S5 (%d)' % PART)
    e('S6 (%d)' % (PART + 1))
    for c in range(1, NCPU):
        lab = g.uniq('comb')
        e('S1 (%d)' % (PART + 2 * c))
        e('S2 S5-FS1')
        e('JP S2,%s' % lab)
        e('S5 S1')
        e('S6 (%d)' % (PART + 2 * c + 1))
        g.lab(lab)
    e('(%d) S6' % (MBOX + 2))
    e('(%d) S5' % (MBOX + 4))
    e('EXIT 2')                      # Foreground: read selected token
    e('J tokloop')

    # ================================================================ barrier
    g.lab('barrier')
    e('S1 [%d]' % LM['bseq'])
    e('S2 %d' % NCPU)
    e('S1 S1+S2')
    e('[%d] S1' % LM['bseq'])
    g.lab('bar_lock')
    e('JSS bar_lock')
    e('S3 (%d)' % BARC)
    e('S2 1')
    e('S3 S3+S2')
    e('(%d) S3' % BARC)
    e('CSM')
    g.lab('bar_wait')
    e('S3 (%d)' % BARC)
    e('S2 S3-S1')
    e('JM S2,bar_wait')
    e('J A6')

    # ================================================================ mv_tile (ternary)
    # In: A1 = tile address. LM mvqt = plane table base, mvgp = #128-groups.
    # Tile layout: G/4 scale vectors (4 x 16-bit fields per row), then per group
    # POS0 POS1 NNEG0 NNEG1.  Scales are decoded once per tile into a Local
    # Memory table; mask vectors are double-buffered through Local Memory.
    # Out: V7 = 64 row results; A1 -> next tile.
    g.lab('mv_tile')
    e('[%d] A6' % LM['link3'])
    e('A6 64')
    e('VL A6')
    e('A4 [%d]' % LM['mvgp'])
    e('[%d] A4' % LM['pcnt'])
    e('A5 A1+A0')                    # scale vectors
    e('A3 16')
    e('A7 A4*A3')
    e('A1 A1+A7')                    # first mask group
    # quad count = G/4 via S shift
    e('S1 A4')
    e('S1 S1>2')
    e('[%d] S1' % LM['tmp1'])
    e('A3 %d' % DTAB)
    e('S5 [%d]' % LM['smask'])
    e('S4 [%d]' % LM['scon'])
    e('A4 11')
    e('A7 16')
    e('A6 32')
    e('[%d] A1' % LM['tmp0'])
    e('A1 64')
    g.lab('mv_dec')
    for line in dec_quad_body()[0]:
        e(line)
    e('S1 [%d]' % LM['tmp1'])
    e('S2 1')
    e('S1 S1-S2')
    e('[%d] S1' % LM['tmp1'])
    e('JN S1,mv_dec')
    e('A1 [%d]' % LM['tmp0'])
    e('A6 64')
    # stage group 0 masks into SLOT0
    e('A7 %d' % SLOT0)
    for k in range(4):
        e('V0 (A1,A2)')
        e('A1 A1+A6')
        e('[A7] V0')
        e('A7 A7+A6')
    e('A4 %d' % SLOT0)
    e('A5 %d' % DTAB)
    e('A3 [%d]' % LM['mvqt'])
    e('V7 V6\\V6')
    e('[A0] V7')                     # Y = 0 in LM 0..63
    g.lab('mv_grp')
    body, cyc = mv_group_body(8, 0, 8, 30)
    for line in body:
        e(line)
    for k in range(4):
        e('A7 A7-A6')
    e('S1 A4')
    e('S2 A7')
    e('A4 S2')
    e('A7 S1')                       # swap mask buffers
    e('S1 [%d]' % LM['pcnt'])
    e('S2 1')
    e('S1 S1-S2')
    e('[%d] S1' % LM['pcnt'])
    e('JN S1,mv_grp')
    for k in range(4):
        e('A1 A1-A6')                # undo the final (unused) prefetch
    e('V7 [A0]')
    e('A6 [%d]' % LM['link3'])
    e('J A6')

    # ================================================================ mv_range_store / add
    # In: A1 = first tile address, A3 = output address, A4 = tiles, A5 = QT, A7 = groups.
    for mode in ('store', 'add'):
        g.lab('mv_range_' + mode)
        e('[%d] A6' % LM['link1'])
        e('[%d] A5' % LM['mvqt'])
        e('[%d] A7' % LM['mvgp'])
        e('A5 [%d]' % LM['cpu'])
        e('A5 A4*A5')                    # first tile index
        e('[%d] A4' % LM['tcount'])
        e('A6 272')
        e('A6 A6*A7')                    # tile words = groups*272
        e('A6 A6*A5')
        e('A1 A1+A6')
        e('A6 64')
        e('A5 A5*A6')
        e('A3 A3+A5')
        e('[%d] A3' % LM['outp'])
        lp = 'mvr_%s_loop' % mode
        g.lab(lp)
        g.call('mv_tile')
        e('A3 [%d]' % LM['outp'])
        if mode == 'store':
            e('(A3,A2) V7')
        else:
            e('V1 (A3,A2)')
            e('V2 V1+FV7')
            e('(A3,A2) V2')
        e('A4 64')
        e('A3 A3+A4')
        e('[%d] A3' % LM['outp'])
        e('A4 [%d]' % LM['tcount'])
        e('A4 A4-A2')
        e('[%d] A4' % LM['tcount'])
        e('JN A4,%s' % lp)
        e('A6 [%d]' % LM['link1'])
        e('J A6')

    def vround24(src, dst, ta, tb):
        e('A4 24')
        e('S1 1')
        e('V%d V%d>A4' % (ta, src))
        e('V%d S1&V%d' % (tb, ta))
        e('S1 [%d]' % LM['c23'])
        e('V%d S1+V%d' % (ta, tb))
        e('V%d V%d+V%d' % (tb, ta, src))
        e('S1 [%d]' % LM['m24'])
        e('V%d S1&V%d' % (ta, tb))
        e('S1 [%d]' % LM['b47'])
        e('V%d S1!V%d' % (tb, ta))
        e('VM V%d,N' % ta)
        e('V%d V%d!V%d&VM' % (dst, tb, ta))

    def sround24(r):  # scalar S<r>, uses S2,S3
        lab = g.uniq('r24')
        e('S2 S%d' % r)
        e('S2 S2>24')
        e('S3 1')
        e('S2 S2&S3')
        e('S3 [%d]' % LM['c23'])
        e('S2 S2+S3')
        e('S%d S%d+S2' % (r, r))
        e('S3 [%d]' % LM['m24'])
        e('S%d S%d&S3' % (r, r))
        e('S3 [%d]' % LM['coefm'])
        e('S3 S%d&S3' % r)
        e('JN S3,%s' % lab)
        e('S3 [%d]' % LM['b47'])
        e('S%d S%d!S3' % (r, r))
        g.lab(lab)

    def vrne(y, q):  # q = round-half-even(y) as integer; clobbers V1-V4, S1,S2,S7
        e('V1 FIX,V%d' % y)
        e('V2 FLT,V1')
        e('V3 V%d-FV2' % y)
        e('S1 [%d]' % LM['absm'])
        e('V4 S1&V3')
        e('S1 [%d]' % LM['half'])
        e('V3 S1-FV4')
        e('VM V3,Z')
        e('S7 VM')
        e('S1 1')
        e('V3 S1&V1')
        e('VM V3,N')
        e('S2 VM')
        e('S7 S7&S2')
        e('S1 -1')
        e('V2 S1+V1')
        e('S1 1')
        e('V3 S1+V1')
        e('VM V1,P')
        e('V4 V2!V3&VM')
        e('VM S7')
        e('V%d V4!V1&VM' % q)

    # quantk: native quantize_row_q8_K over one 256-block, F32 rounding reproduced.
    # In: A1 = 256 floats (CM), A3 = QT entry of the first 128-group.
    # Out: two entries (stride QTW): planes [w*8+p] (w = 64-column half, p = bit),
    # [16] K1 = d, [17] K2 = -T*d (T = sum of codes of the 128-group).
    # A1 += 256, A3 += 2*QTW.
    # ================================================================ normq4096
    # In: A1 = source (4096 floats), A3 = QT base, LM wn = norm weights.
    # RMSNorm with ggml's F32 rounding: sum of F32(x*x) (wide sum), mean=F32(sum/n),
    # s=F32(sqrt(F32(mean+eps))), scale=F32(1/s), y=F32(F32(x*scale)*w).
    # Every CPU forms the full sum; each writes XN for its own 8 groups and
    # quantizes them into QT.
    g.lab('normq4096')
    e('[%d] A6' % LM['link1'])
    e('[%d] A3' % LM['qtp'])
    e('[%d] A1' % LM['src'])
    e('A3 64')
    e('VL A3')
    e('V5 V6\\V6')
    e('A5 32')
    g.lab('nq_ss')                    # 64 chunks, 2 per iteration
    for acc, nacc in ((5, 7), (7, 5)):
        e('V1 (A1,A2)')
        e('A1 A1+A3')
        e('V2 V1*FV1')
        vround24(2, 3, 4, 6)
        e('V%d V%d+FV3' % (nacc, acc))
    e('A5 A5-A2')
    e('JN A5,nq_ss')
    g.hsum(5)
    e('S2 [%d]' % LM['invh'])
    e('S6 S1*FS2')
    sround24(6)                       # mean
    e('S2 [%d]' % LM['eps'])
    e('S6 S6+FS2')
    sround24(6)                       # mean + eps
    g.rsqrt(6, 1)
    e('S1 S6*FS1')
    sround24(1)                       # sqrtf
    g.recip(1, 6)
    sround24(6)                       # scale = 1/s
    e('A4 [%d]' % LM['cpu'])
    e('A3 1024')
    e('A4 A4*A3')
    e('A1 [%d]' % LM['src'])
    e('A1 A1+A4')
    e('S1 [%d]' % LM['wn'])
    e('A5 S1')
    e('A5 A5+A4')                     # norm weights
    e('A7 %d' % XN)
    e('A7 A7+A4')
    e('A3 64')
    e('A6 16')
    g.lab('nq_scale')
    e('V1 (A1,A2)')
    e('V2 S6*FV1')
    vround24(2, 3, 4, 5)
    e('V7 (A5,A2)')
    e('V2 V3*FV7')
    vround24(2, 1, 4, 5)
    e('(A7,A2) V1')
    e('A1 A1+A3')
    e('A5 A5+A3')
    e('A7 A7+A3')
    e('A6 A6-A2')
    e('JN A6,nq_scale')
    # quantize the 8 groups
    e('A4 [%d]' % LM['cpu'])
    e('A3 1024')
    e('A4 A4*A3')
    e('A1 %d' % XN)
    e('A1 A1+A4')
    e('A4 [%d]' % LM['cpu'])
    e('A3 512')
    e('A4 A4*A3')                     # 8 groups * 16 words
    e('A3 [%d]' % LM['qtp'])
    e('A3 A3+A4')
    e('A4 4')
    g.lab('nq_q')
    e('[%d] A4' % LM['n'])
    g.call('quantk')
    e('A4 [%d]' % LM['n'])
    e('A4 A4-A2')
    e('JN A4,nq_q')
    e('A6 [%d]' % LM['link1'])
    e('J A6')

    # ================================================================ quantk
    # Native quantize_row_q8_K over one 256-element block (iscale = -QMAX/max),
    # reproducing IEEE-F32 rounding of x, of the scale and of x*iscale, and
    # round-half-even to integer, with CRAY-2 integer/logical vector operations.
    # In: A1 = 256 floats, A3 = QT entry of the first 128-group.  Out: two
    # entries (stride QTW) of bit-plane words and K1 = d, K2 = -T*d;
    # A1 += 256, A3 += 2*QTW.
    g.lab('quantk')
    e('[%d] A6' % LM['link2'])
    e('[%d] A3' % LM['sc'])
    e('A4 64')
    e('VL A4')
    e('S5 [%d]' % LM['absm'])
    for v in range(4):
        e('V1 (A1,A2)')
        e('A4 64')
        e('A1 A1+A4')
        vround24(1, 3, 5, 6)               # x as IEEE F32
        e('A4 %d' % (LM['xq'] + 64 * v))
        e('[A4] V3')
        e('S5 [%d]' % LM['absm'])
        if v == 0:
            e('V7 S5&V3')
        else:
            e('V4 S5&V3')
            e('V5 V7-FV4')
            e('VM V5,P')
            e('V6 V7!V4&VM')
            e('V7 V6!V6')
    e('[%d] A1' % LM['tmp0'])
    g.hmax(7, None)                         # S1 = amax
    e('A3 [%d]' % LM['sc'])
    e('JZ S1,qk_zero')
    # signed max: first element (index order) with |x| == amax
    e('S5 [%d]' % LM['absm'])
    for v in range(4):
        e('A4 %d' % (LM['xq'] + 64 * v))
        e('V1 [A4]')
        e('V2 S5&V1')
        e('V3 S1-FV2')
        e('VM V3,Z')
        e('S2 VM')
        e('JN S2,qk_found')
    e('ERR')                                # unreachable
    g.lab('qk_found')
    e('S3 ZS2')
    e('A5 S3')
    e('A4 A4+A5')
    e('S6 [A4]')                            # max (signed)
    # iscale = F32(-127/max); d = F32(1/iscale)
    g.recip(6, 5)                           # S5 = 1/max (uses S3,S4)
    e('S2 %d' % cf(-float(QMAX)))
    e('S6 S2*FS5')
    sround24(6)
    e('[%d] S6' % LM['iscl'])
    g.recip(6, 5)
    e('S6 S5')
    sround24(6)
    e('[%d] S6' % LM['sval'])               # d
    for v in range(4):
        h, w = divmod(v, 2)
        if w == 0:
            e('[%d] S0' % LM['tacc'])
        e('A4 %d' % (LM['xq'] + 64 * v))
        e('V1 [A4]')
        e('S6 [%d]' % LM['iscl'])
        e('V3 S6*FV1')
        vround24(3, 5, 1, 4)                # F32(iscale*x)
        vrne(5, 6)                          # nearest_int -> V6
        e('S4 [%d]' % LM['tacc'])
        for p in range(NPL):
            e('S1 %d' % (1 << p))
            e('V3 S1&V6')
            e('VM V3,N')
            e('S2 VM')
            e('(A3,%d) S2' % (h * QTW + w * 8 + p))
            e('S3 PS2')
            if p:
                e('S3 S3<%d' % p)
            e('S4 S4%sS3' % ('-' if p == NPL - 1 else '+'))
        e('[%d] S4' % LM['tacc'])
        if w == 1:
            e('S1 [%d]' % LM['sval'])
            e('(A3,%d) S1' % (h * QTW + 16))    # K1 = d
            e('S4 FLT,S4')
            e('S4 S1*FS4')
            e('S4 S0-FS4')
            e('(A3,%d) S4' % (h * QTW + 17))    # K2 = -T d
    e('J qk_done')
    g.lab('qk_zero')
    for q in range(18):
        e('(A3,%d) S0' % q)
        e('(A3,%d) S0' % (QTW + q))
    g.lab('qk_done')
    e('A1 [%d]' % LM['tmp0'])
    e('A4 %d' % (2 * QTW))
    e('A3 A3+A4')
    e('A6 [%d]' % LM['link2'])
    e('J A6')

    # ================================================================ vexp
    # V1 -> V2 = exp(V1).  Clobbers V3-V5, S1-S2, A4.
    g.lab('vexp')
    e('S1 [%d]' % LM['log2e'])
    e('V3 S1*FV1')                    # t
    e('V4 FIX,V3')                    # k (rounded)
    e('V5 FLT,V4')
    e('V2 V3-FV5')                    # r in [-.5,.5]
    e('S1 [%d]' % LM['ln2'])
    e('V3 S1*FV2')                    # y
    e('S1 [%d]' % LM['expc'])
    e('S2 [%d]' % (LM['expc'] + 1))
    e('V5 S1*FV3')
    e('V2 S2+FV5')
    for i in range(2, EXPDEG + 1):
        e('S2 [%d]' % (LM['expc'] + i))
        e('V5 V2*FV3')
        e('V2 S2+FV5')
    e('A4 48')
    e('V5 V4<A4')
    e('V3 V2+V5')                     # scale by 2^k via exponent add
    e('V2 V3!V3')                     # move result to V2 (copy)
    e('J A6')

    # ================================================================ ffn_gateup
    # CPU c: tiles 48c..48c+47 of gate and up. h = silu(g)*u -> H; quantize 24 groups -> QTB
    g.lab('ffn_gateup')
    e('[%d] A6' % LM['link1'])
    e('A1 %d' % QTA)
    e('[%d] A1' % LM['mvqt'])
    e('A1 32')
    e('[%d] A1' % LM['mvgp'])
    e('A4 [%d]' % LM['cpu'])
    e('A3 48')
    e('A4 A4*A3')                     # first tile
    e('A3 %d' % tile_w(HID))
    e('A3 A4*A3')
    e('A5 [%d]' % LM['desc'])
    e('S1 (A5,2)')
    e('A1 S1')
    e('A1 A1+A3')
    e('[%d] A1' % LM['gptr'])
    e('S1 (A5,3)')
    e('A1 S1')
    e('A1 A1+A3')
    e('[%d] A1' % LM['uptr'])
    e('A3 64')
    e('A4 A4*A3')
    e('A3 %d' % H)
    e('A3 A3+A4')
    e('[%d] A3' % LM['hptr'])
    e('A4 48')
    e('[%d] A4' % LM['tcount'])
    g.lab('fg_loop')
    e('A1 [%d]' % LM['gptr'])
    g.call('mv_tile')
    e('[%d] A1' % LM['gptr'])
    e('A3 %d' % LM['gt'])
    e('[A3] V7')                      # gate -> LM
    e('A1 [%d]' % LM['uptr'])
    g.call('mv_tile')
    e('[%d] A1' % LM['uptr'])
    e('A3 %d' % LM['gt'])
    e('V6 [A3]')                      # g ; V7 = u
    e('V1 S0-FV6')                    # -g
    e('[%d] A6' % LM['link2'])
    g.call('vexp')                    # V2 = exp(-g)
    e('A6 [%d]' % LM['link2'])
    e('S1 [%d]' % LM['one'])
    e('V3 S1+FV2')                    # 1 + exp(-g)
    e('V4 /HV3')
    e('V5 V4*IV3')
    e('V1 V4*FV5')                    # 1/(1+exp(-g))
    e('V2 V6*FV1')                    # silu(g)
    e('V3 V2*FV7')                    # * u
    e('A3 [%d]' % LM['hptr'])
    e('(A3,A2) V3')
    e('A4 64')
    e('A3 A3+A4')
    e('[%d] A3' % LM['hptr'])
    e('A4 [%d]' % LM['tcount'])
    e('A4 A4-A2')
    e('[%d] A4' % LM['tcount'])
    e('JN A4,fg_loop')
    # quantize own 24 groups of H
    e('A4 [%d]' % LM['cpu'])
    e('A3 3072')
    e('A4 A4*A3')
    e('A1 %d' % H)
    e('A1 A1+A4')
    e('A4 [%d]' % LM['cpu'])
    e('A3 1536')
    e('A4 A4*A3')
    e('A3 %d' % QTB)
    e('A3 A3+A4')
    e('A4 12')
    g.lab('fg_q')
    e('[%d] A4' % LM['n'])
    g.call('quantk')
    e('A4 [%d]' % LM['n'])
    e('A4 A4-A2')
    e('JN A4,fg_q')
    e('A6 [%d]' % LM['link1'])
    e('J A6')

    # ================================================================ headnorm
    # qk-norm + RoPE of one 128-vector.  In: A1 = source (CM), A5 = norm weights (CM).
    # Out: V1 (dims 0..63), V2 (dims 64..127).  LM rope = rope table address.
    g.lab('headnorm')
    e('A4 64')
    e('V1 (A1,A2)')
    e('A1 A1+A4')
    e('V2 (A1,A2)')
    e('V3 V1*FV1')
    e('V4 V2*FV2')
    e('V5 V3+FV4')
    e('[%d] A6' % LM['link3'])
    e('[%d] A5' % LM['tmp1'])
    e('[%d] A1' % LM['tmp0'])
    g.hsum(5)
    e('S2 %d' % cf(1.0 / HD))
    e('S1 S1*FS2')
    e('S2 [%d]' % LM['eps'])
    e('S1 S1+FS2')
    g.rsqrt(1, 6)
    e('A1 [%d]' % LM['tmp0'])
    e('A4 64')
    e('A1 A1-A4')
    e('V1 (A1,A2)')
    e('A1 A1+A4')
    e('V2 (A1,A2)')
    e('A5 [%d]' % LM['tmp1'])
    e('V3 (A5,A2)')
    e('A5 A5+A4')
    e('V4 (A5,A2)')
    e('V5 S6*FV1')
    e('V1 V5*FV3')                    # normed x0
    e('V5 S6*FV2')
    e('V2 V5*FV4')                    # normed x1
    e('A3 [%d]' % LM['rope'])
    e('V3 (A3,A2)')                   # cos
    e('A3 A3+A4')
    e('V4 (A3,A2)')                   # sin
    e('V5 V1*FV3')
    e('V6 V2*FV4')
    e('V7 V5-FV6')                    # x0 cos - x1 sin
    e('V5 V1*FV4')
    e('V6 V2*FV3')
    e('V2 V5+FV6')                    # x0 sin + x1 cos
    e('V1 V7!V7')
    e('A6 [%d]' % LM['link3'])
    e('J A6')

    # ================================================================ attention
    g.lab('attention')
    e('[%d] A6' % LM['link1'])
    e('A3 64')
    e('VL A3')
    # rope table for pos; ctx n = pos+1
    e('S1 [%d]' % LM['pos'])
    e('A1 S1')
    e('A3 128')
    e('A1 A1*A3')
    e('A3 %d' % ROPE)
    e('A1 A1+A3')
    e('[%d] A1' % LM['rope'])
    e('A1 S1')
    e('A1 A1+A2')
    e('[%d] A1' % LM['n'])
    # layer cache bases
    e('A4 [%d]' % LM['desc'])
    e('S1 (A4,9)')
    e('[%d] S1' % LM['ktb'])
    e('S1 (A4,10)')
    e('[%d] S1' % LM['vtb'])
    e('A4 [%d]' % LM['cpu'])
    e('A4 A4+A4')
    e('[%d] A4' % LM['kh'])            # first kv head = 2*cpu
    e('A4 2')
    e('[%d] A4' % LM['h'])             # kv heads remaining
    g.lab('att_kv')
    # --- K: norm, rope, store transposed at column pos
    e('A1 [%d]' % LM['kh'])
    e('A3 %d' % HD)
    e('A1 A1*A3')
    e('[%d] A1' % LM['kvoff'])
    e('A3 %d' % (QKV + HID))
    e('A1 A1+A3')
    e('A4 [%d]' % LM['desc'])
    e('S1 (A4,8)')
    e('A5 S1')
    g.call('headnorm')
    e('A1 [%d]' % LM['kh'])
    e('A3 %d' % (HD * CTXS))
    e('A1 A1*A3')
    e('S1 [%d]' % LM['ktb'])
    e('A3 S1')
    e('A1 A1+A3')
    e('S1 [%d]' % LM['pos'])
    e('A3 S1')
    e('A1 A1+A3')                     # K^T(kh) + pos
    e('A3 %d' % CTXS)
    e('(A1,A3) V1')
    e('A4 %d' % (64 * CTXS))
    e('A1 A1+A4')
    e('(A1,A3) V2')
    # --- V: store row pos
    e('A1 [%d]' % LM['kvoff'])
    e('A3 %d' % (QKV + HID + NKV * HD))
    e('A1 A1+A3')
    e('V1 (A1,A2)')
    e('A4 64')
    e('A1 A1+A4')
    e('V2 (A1,A2)')
    e('S1 [%d]' % LM['pos'])
    e('A3 S1')
    e('A4 %d' % (NKV * HD))
    e('A3 A3*A4')
    e('A4 [%d]' % LM['kvoff'])
    e('A3 A3+A4')
    e('S1 [%d]' % LM['vtb'])
    e('A4 S1')
    e('A3 A3+A4')
    e('(A3,A2) V1')
    e('A4 64')
    e('A3 A3+A4')
    e('(A3,A2) V2')
    # --- 4 query heads: norm, rope, scale -> LM qh
    for hq in range(4):
        e('A1 [%d]' % LM['kh'])
        e('A3 4')
        e('A1 A1*A3')
        if hq:
            e('A3 %d' % hq)
            e('A1 A1+A3')
        e('A3 %d' % HD)
        e('A1 A1*A3')
        e('A3 %d' % QKV)
        e('A1 A1+A3')
        e('A4 [%d]' % LM['desc'])
        e('S1 (A4,7)')
        e('A5 S1')
        g.call('headnorm')
        e('S1 [%d]' % LM['hdsc'])
        e('V3 S1*FV1')
        e('V4 S1*FV2')
        e('A3 %d' % (LM['qh'] + hq * HD))
        e('[A3] V3')
        e('A3 %d' % (LM['qh'] + hq * HD + 64))
        e('[A3] V4')
    # --- scores: for head pairs (0,1),(2,3); chunks of 64 positions
    for pair in (0, 1):
        ha, hb = 2 * pair, 2 * pair + 1
        e('A1 0')
        e('[%d] A1' % LM['grp'])           # chunk start
        lc = g.uniq('sc_chunk')
        g.lab(lc)
        # VL = min(64, n - start)
        e('A1 [%d]' % LM['n'])
        e('A3 [%d]' % LM['grp'])
        e('A1 A1-A3')
        e('A4 64')
        e('A4 A4-A1')
        lv = g.uniq('vl')
        e('JM A4,%s' % lv)
        e('VL A1')
        e('J %s_x' % lv)
        g.lab(lv)
        e('A4 64')
        e('VL A4')
        g.lab(lv + '_x')
        # K^T(kh) + start
        e('A1 [%d]' % LM['kh'])
        e('A4 %d' % (HD * CTXS))
        e('A1 A1*A4')
        e('S1 [%d]' % LM['ktb'])
        e('A4 S1')
        e('A1 A1+A4')
        e('A1 A1+A3')
        e('A3 %d' % (LM['qh'] + ha * HD))
        e('A4 %d' % (LM['qh'] + hb * HD))
        e('A5 %d' % CTXS)
        e('V4 V3\\V3')
        e('V6 V3\\V3')
        e('A7 64')
        ld = g.uniq('sc_d')
        g.lab(ld)
        for half in (0, 1):
            acc_a, nacc_a = (4, 5) if half == 0 else (5, 4)
            acc_b, nacc_b = (6, 7) if half == 0 else (7, 6)
            e('V1 (A1,A2)')
            e('A1 A1+A5')
            e('S1 [A3]')
            e('A3 A3+A2')
            e('S2 [A4]')
            e('A4 A4+A2')
            e('V2 S1*FV1')
            e('V3 S2*FV1')
            e('V%d V%d+FV2' % (nacc_a, acc_a))
            e('V%d V%d+FV3' % (nacc_b, acc_b))
        e('A7 A7-A2')
        e('JN A7,%s' % ld)
        # store score chunks to LM scores + head*1024 + start
        e('A3 [%d]' % LM['grp'])
        e('A4 %d' % (LM['scores'] + ha * CTX))
        e('A4 A4+A3')
        e('[A4] V4')
        e('A4 %d' % (LM['scores'] + hb * CTX))
        e('A4 A4+A3')
        e('[A4] V6')
        e('A4 64')
        e('A3 A3+A4')
        e('[%d] A3' % LM['grp'])
        e('A1 [%d]' % LM['n'])
        e('A1 A3-A1')
        e('JM A1,%s' % lc)
    # --- softmax per head (4 heads), in LM scores
    for hq in range(4):
        base = LM['scores'] + hq * CTX
        # max: running vector max over chunks, padded lanes use -inf
        e('S1 [%d]' % LM['ninf'])
        e('V7 V6\\V6')
        e('V6 S1+FV7')                     # M = -inf
        e('A3 %d' % base)
        e('[%d] A3' % LM['grp'])
        lm_ = g.uniq('smx')
        g.lab(lm_)
        g.set_vl_rem('A3', base)
        e('V7 V6\\V6')
        e('A4 64')
        e('VL A4')
        e('V5 S1+FV7')                     # -inf fill (full 64)
        e('VL A1')
        e('V5 [A3]')                       # partial chunk overwrites low lanes
        e('A4 64')
        e('VL A4')
        e('V2 V6-FV5')
        e('VM V2,M')                       # M < x
        e('V3 V5!V6&VM')
        e('V6 V3!V3')
        e('A4 64')
        e('A3 A3+A4')
        e('[%d] A3' % LM['grp'])
        e('A1 %d' % (base + 0))
        e('A4 [%d]' % LM['n'])
        e('A1 A1+A4')
        e('A1 A3-A1')
        e('JM A1,%s' % lm_)
        g.hmax(6, None)                    # S1 = max
        e('[%d] S1' % LM['tmp0'])
        # exp(s - max), sum
        e('V7 V6\\V6')
        e('[%d] S0' % LM['tmp1'])
        e('A3 %d' % base)
        e('[%d] A3' % LM['grp'])
        le = g.uniq('sexp')
        g.lab(le)
        g.set_vl_rem('A3', base)
        e('V6 [A3]')
        e('S1 [%d]' % LM['tmp0'])
        e('V5 V6\\V6')
        e('V7 S1+FV5')
        e('V1 V6-FV7')                     # s - max
        e('[%d] A6' % LM['link2'])
        g.call('vexp')
        e('A6 [%d]' % LM['link2'])
        e('A3 [%d]' % LM['grp'])
        e('[A3] V2')
        # sum of the VL elements: zero-pad to 64 then hsum later; accumulate in LM scalar
        e('A5 %d' % LM['lmt2'])
        e('A4 64')
        e('VL A4')
        e('V3 V6\\V6')
        e('[A5] V3')
        e('VL A1')
        e('[A5] V2')
        e('A4 64')
        e('VL A4')
        e('V5 [A5]')
        g.hsum(5)
        e('S2 [%d]' % LM['tmp1'])
        e('S2 S2+FS1')
        e('[%d] S2' % LM['tmp1'])
        e('A3 [%d]' % LM['grp'])
        e('A4 64')
        e('A3 A3+A4')
        e('[%d] A3' % LM['grp'])
        e('A1 %d' % base)
        e('A4 [%d]' % LM['n'])
        e('A1 A1+A4')
        e('A1 A3-A1')
        e('JM A1,%s' % le)
        e('S1 [%d]' % LM['tmp1'])
        g.recip(1, 2)
        e('[%d] S2' % (LM['qh'] + hq * HD))  # 1/sum (q no longer needed)
    # --- weighted V sum: for chunk c (dims), head pairs, over positions t
    for pair in (0, 1):
        ha, hb = 2 * pair, 2 * pair + 1
        for ch in (0, 1):
            e('A3 64')
            e('VL A3')
            e('A1 [%d]' % LM['kvoff'])
            e('S1 [%d]' % LM['vtb'])
            e('A4 S1')
            e('A1 A1+A4')
            if ch:
                e('A4 64')
                e('A1 A1+A4')
            e('A3 %d' % (LM['scores'] + ha * CTX))
            e('A4 %d' % (LM['scores'] + hb * CTX))
            e('A5 %d' % (NKV * HD))
            e('A7 [%d]' % LM['n'])
            e('V4 V3\\V3')
            e('V6 V3\\V3')
            # n may be odd: handle in pairs with a tail
            lt = g.uniq('pv')
            lt1 = lt + '_one'
            ltd = lt + '_done'
            g.lab(lt)
            e('A7 A7-A2')
            e('JM A7,%s' % ltd)
            # first of pair
            e('V1 (A1,A2)')
            e('A1 A1+A5')
            e('S1 [A3]')
            e('A3 A3+A2')
            e('S2 [A4]')
            e('A4 A4+A2')
            e('V2 S1*FV1')
            e('V3 S2*FV1')
            e('V5 V4+FV2')
            e('V7 V6+FV3')
            e('A7 A7-A2')
            e('JM A7,%s' % lt1)
            e('V1 (A1,A2)')
            e('A1 A1+A5')
            e('S1 [A3]')
            e('A3 A3+A2')
            e('S2 [A4]')
            e('A4 A4+A2')
            e('V2 S1*FV1')
            e('V3 S2*FV1')
            e('V4 V5+FV2')
            e('V6 V7+FV3')
            e('J %s' % lt)
            g.lab(lt1)
            e('V4 V5!V5')
            e('V6 V7!V7')
            g.lab(ltd)
            for hh, vr in ((ha, 4), (hb, 6)):
                e('S1 [%d]' % (LM['qh'] + hh * HD))
                e('V2 S1*FV%d' % vr)
                e('A1 [%d]' % LM['kh'])
                e('A3 4')
                e('A1 A1*A3')
                if hh:
                    e('A3 %d' % hh)
                    e('A1 A1+A3')
                e('A3 %d' % HD)
                e('A1 A1*A3')
                e('A3 %d' % (ATT + 64 * ch))
                e('A1 A1+A3')
                e('(A1,A2) V2')
    # next kv head
    e('A1 [%d]' % LM['kh'])
    e('A1 A1+A2')
    e('[%d] A1' % LM['kh'])
    e('A1 [%d]' % LM['h'])
    e('A1 A1-A2')
    e('[%d] A1' % LM['h'])
    e('JN A1,att_kv')
    # quantize own 8 heads of ATT -> QTB groups 8c..8c+7
    e('A4 [%d]' % LM['cpu'])
    e('A3 1024')
    e('A4 A4*A3')
    e('A1 %d' % ATT)
    e('A1 A1+A4')
    e('A4 [%d]' % LM['cpu'])
    e('A3 512')
    e('A4 A4*A3')
    e('A3 %d' % QTB)
    e('A3 A3+A4')
    e('A4 4')
    g.lab('att_q')
    e('[%d] A4' % LM['n'])
    g.call('quantk')
    e('A4 [%d]' % LM['n'])
    e('A4 A4-A2')
    e('JN A4,att_q')
    e('A6 [%d]' % LM['link1'])
    e('J A6')

    # ================================================================ head
    head = T['output.weight']['addr']
    per = (HEAD_TILES + NCPU - 1) // NCPU     # 593
    g.lab('head')
    e('[%d] A6' % LM['link1'])
    e('A1 %d' % QTA)
    e('[%d] A1' % LM['mvqt'])
    e('A1 32')
    e('[%d] A1' % LM['mvgp'])
    e('A4 [%d]' % LM['cpu'])
    e('A3 %d' % per)
    e('A4 A4*A3')                     # first tile
    e('[%d] A4' % LM['tile'])
    e('A3 %d' % tile_w(HID))
    e('A3 A4*A3')
    e('A1 %d' % head)
    e('A1 A1+A3')
    e('[%d] A1' % LM['wptr'])
    # tiles for this CPU: min(per, HEAD_TILES - first)
    e('A3 %d' % HEAD_TILES)
    e('A3 A3-A4')
    e('A5 %d' % per)
    e('A5 A3-A5')
    e('JP A5,hd_full')
    e('[%d] A3' % LM['tcount'])
    e('J hd_init')
    g.lab('hd_full')
    e('A5 %d' % per)
    e('[%d] A5' % LM['tcount'])
    g.lab('hd_init')
    e('A3 64')
    e('VL A3')
    e('S1 [%d]' % LM['ninf'])
    e('V7 V6\\V6')
    e('V1 S1+FV7')
    e('A3 %d' % LM['am'])
    e('[A3] V1')
    e('A3 %d' % LM['ai'])
    e('[A3] V7')
    g.lab('hd_loop')
    e('A1 [%d]' % LM['wptr'])
    g.call('mv_tile')
    e('[%d] A1' % LM['wptr'])
    e('A4 [%d]' % LM['tile'])
    e('A3 64')
    e('A3 A4*A3')
    e('A5 %d' % LOGITS)
    e('A5 A5+A3')
    e('(A5,A2) V7')                   # logits
    e('S2 A3')                        # row base
    # last tile: mask padded rows with -inf
    e('A5 %d' % (HEAD_TILES - 1))
    e('A5 A4-A5')
    e('JN A5,hd_nomask')
    e('S1 [%d]' % LM['ninf'])
    e('V5 V6\\V6')
    e('V6 S1+FV5')
    valid = VOCAB - (HEAD_TILES - 1) * 64
    e('S3 %d' % (((1 << valid) - 1) << (64 - valid)))
    e('VM S3')
    e('V5 V7!V6&VM')
    e('V7 V5!V5')
    g.lab('hd_nomask')
    e('A3 %d' % LM['am'])
    e('V1 [A3]')                      # M
    e('A3 %d' % LM['ai'])
    e('V2 [A3]')                      # I
    e('A3 %d' % LM['iota'])
    e('V3 [A3]')
    e('V4 S2+V3')                     # row indices
    e('V5 V1-FV7')
    e('VM V5,M')                      # M < Y
    e('V6 V7!V1&VM')
    e('V3 V4!V2&VM')
    e('A3 %d' % LM['am'])
    e('[A3] V6')
    e('A3 %d' % LM['ai'])
    e('[A3] V3')
    e('A4 [%d]' % LM['tile'])
    e('A4 A4+A2')
    e('[%d] A4' % LM['tile'])
    e('A4 [%d]' % LM['tcount'])
    e('A4 A4-A2')
    e('[%d] A4' % LM['tcount'])
    e('JN A4,hd_loop')
    g.argmax_lanes('hd_lane')                     # -> S5 best logit, S6 its token
    e('A4 [%d]' % LM['cpu'])
    e('A4 A4+A4')
    e('A3 %d' % PART)
    e('A4 A4+A3')
    e('(A4) S5')
    e('A3 1')
    e('A4 A4+A3')
    e('(A4) S6')
    e('A6 [%d]' % LM['link1'])
    e('J A6')

    # ================================================================ data
    # per-layer descriptors
    data = {}
    for l in range(L):
        d = [0] * DESC_W
        d[0] = T['blk.%d.attn_q.weight' % l]['addr']
        d[1] = T['blk.%d.attn_output.weight' % l]['addr']
        d[2] = T['blk.%d.ffn_gate.weight' % l]['addr']
        d[3] = T['blk.%d.ffn_up.weight' % l]['addr']
        d[4] = T['blk.%d.ffn_down.weight' % l]['addr']
        d[5] = T['blk.%d.attn_norm.weight' % l]['addr']
        d[6] = T['blk.%d.ffn_norm.weight' % l]['addr']
        d[7] = T['blk.%d.attn_q_norm.weight' % l]['addr']
        d[8] = T['blk.%d.attn_k_norm.weight' % l]['addr']
        d[9] = KVBASE + l * KV_L
        d[10] = KVBASE + l * KV_L + KT_L
        assert T['blk.%d.attn_k.weight' % l]['addr'] == d[0] + 64 * tile_w(HID)
        assert T['blk.%d.attn_v.weight' % l]['addr'] == d[0] + 80 * tile_w(HID)
        data[DESC + l * DESC_W] = d
    meta = dict(EMB=EMB, QTW=QTW, MBOX=MBOX, X=X, XN=XN, QKV=QKV, ATT=ATT, H=H, QTA=QTA, QTB=QTB, LOGITS=LOGITS,
                ROPE=ROPE, DESC=DESC, KVBASE=KVBASE, CTX=CTX, CTXS=CTXS, KT_L=KT_L, V_L=V_L)
    return '\n'.join(g.out) + '\n', data, meta


def dec_quad_body():
    """Decode one scale vector (4 groups) into the LM table.  A5 -> scale vector
    (advanced by 64 via A2*64 trick is not available, so via S), A3 -> LM table
    slot (advanced by 256), A4 = 11, A7 = 16, A6 = 32; S4/S5 = constants."""
    from .sched import Op, schedule
    ops = []
    vld = Op('VLD', '{vS} (A5,A2)', dst='vS')
    ops.append(vld)
    ops.append(Op('A', 'A5 A5+A1', after=[vld]))
    shifts = {1: 'A7', 2: 'A6'}
    outs = []
    for q in range(4):
        if q == 0:
            ops.append(Op('I', '{v2_0} {vS}>A4', dst='v2_0', srcs=['vS']))
        elif q in shifts:
            ops.append(Op('I', '{v1_%d} {vS}<%s' % (q, shifts[q]), dst='v1_%d' % q, srcs=['vS']))
            ops.append(Op('I', '{v2_%d} {v1_%d}>A4' % (q, q), dst='v2_%d' % q, srcs=['v1_%d' % q]))
        else:  # q == 3: shift 48 = 32 + 16
            ops.append(Op('I', '{v1a_3} {vS}<A6', dst='v1a_3', srcs=['vS']))
            ops.append(Op('I', '{v1_3} {v1a_3}<A7', dst='v1_3', srcs=['v1a_3']))
            ops.append(Op('I', '{v2_3} {v1_3}>A4', dst='v2_3', srcs=['v1_3']))
        ops.append(Op('L', '{v3_%d} S5&{v2_%d}' % (q, q), dst='v3_%d' % q, srcs=['v2_%d' % q]))
        ops.append(Op('L', '{v4_%d} S4!{v3_%d}' % (q, q), dst='v4_%d' % q, srcs=['v3_%d' % q]))
        outs.append('v4_%d' % q)
    prev = []
    for q in range(4):
        st = Op('VSM', '[A3] {%s}' % outs[q], srcs=[outs[q]], after=prev)
        inc = Op('A', 'A3 A3+A1', after=[st])     # A1 holds 64 during decode
        ops += [st, inc]
        prev = [inc]
    return schedule(ops, vregs=[0, 1, 2, 3, 4, 5, 6, 7], sregs=[1, 2], window=8, reserve=0)      # LM: mask double buffer, decoded scale table


def mv_group_body(window=8, reserve=0, seed=None, jitter=0):
    """One 128-column group with masks already staged in LM (current set at A4
    on entry, slot order POS0 POS1 NNEG0 NNEG1), prefetching the next group's
    masks from CM (A1) into the other set (A7).  D (decoded weight scale) comes
    from the LM table at A5.  On exit A4 is back at its set base, A7 has
    advanced 256, A1 by 256, A5 by 64, A3 by 64."""
    from .sched import Op, schedule
    ops, sld = [], []
    chains = {'A1': [], 'A4': [], 'A7': [], 'A5': []}

    def ch(name, op):
        op.after = list(op.after) + chains[name][-1:]
        chains[name].append(op)
        ops.append(op)
        return op
    cur = [0]

    def goto(off):
        while cur[0] != off:
            if off > cur[0]:
                ch('A4', Op('A', 'A4 A4+A6'))
                cur[0] += 64
            else:
                ch('A4', Op('A', 'A4 A4-A6'))
                cur[0] -= 64
    accs = []
    pf = 0
    for ph, slots in (('p', (0, 64)), ('n', (128, 192))):
        acc = None
        k = 0
        for p in range(NPL - 1, -1, -1):
            # interleave one prefetch copy every 4 planes
            if p in (NPL - 2, 2) and pf < 4:
                v = 'vpf%d' % pf
                ch('A1', Op('VLD', '{%s} (A1,A2)' % v, dst=v))
                ch('A1', Op('A', 'A1 A1+A6'))
                ch('A7', Op('VSM', '[A7] {%s}' % v, srcs=[v]))
                ch('A7', Op('A', 'A7 A7+A6'))
                pf += 1
            t = 'sT%s%d' % (ph, p)
            ops.append(Op('SLM', '{%s} [%d]' % (t, LM['tags'] + p), dst=t, parcels=2))
            order = (0, 1) if cur[0] != slots[1] else (1, 0)
            for w in order:
                pl, wv, a, b, c = ['%s%s%d%d' % (x, ph, p, w) for x in ('sP', 'vw', 'va', 'vb', 'vc')]
                o = Op('SLD', '{%s} (A3,%d)' % (pl, w * 8 + p), dst=pl, parcels=3)
                sld.append(o)
                ops.append(o)
                goto(slots[w])
                ch('A4', Op('VLM', '{%s} [A4]' % wv, dst=wv))
                ops.append(Op('L', '{%s} {%s}&{%s}' % (a, pl, wv), dst=a, srcs=[pl, wv]))
                ops.append(Op('I', '{%s} P{%s}' % (b, a), dst=b, srcs=[a]))
                if k % 2 == 0:
                    ops.append(Op('L', '{%s} {%s}!{%s}' % (c, t, b), dst=c, srcs=[t, b]))
                else:
                    ops.append(Op('I', '{%s} {%s}+{%s}' % (c, t, b), dst=c, srcs=[t, b]))
                nacc = 'vacc%s%d' % (ph, k)
                k += 1
                if acc is None:
                    ops.append(Op('F', '{%s} S0+F{%s}' % (nacc, c), dst=nacc, srcs=[c]))
                else:
                    ops.append(Op('F', '{%s} {%s}+F{%s}' % (nacc, acc, c), dst=nacc, srcs=[acc, c]))
                acc = nacc
        accs.append(acc)
    goto(0)
    ops.append(Op('F', '{vsum} {%s}+F{%s}' % tuple(accs), dst='vsum', srcs=list(accs)))
    for kk, off in (('sK1', 16), ('sK2', 17)):
        o = Op('SLD', '{%s} (A3,%d)' % (kk, off), dst=kk, parcels=3)
        sld.append(o)
        ops.append(o)
    ops.append(Op('M', '{vm1} {sK1}*F{vsum}', dst='vm1', srcs=['sK1', 'vsum']))
    ops.append(Op('F', '{vf1} {sK2}+F{vm1}', dst='vf1', srcs=['sK2', 'vm1']))
    ch('A5', Op('VLM', '{vD} [A5]', dst='vD'))
    ch('A5', Op('A', 'A5 A5+A6'))
    ops.append(Op('M', '{vm2} {vf1}*F{vD}', dst='vm2', srcs=['vf1', 'vD']))
    ops.append(Op('VLM', '{vyl} [A0]', dst='vyl'))
    ops.append(Op('F', '{vY} {vyl}+F{vm2}', dst='vY', srcs=['vyl', 'vm2']))
    ops.append(Op('VSM', '[A0] {vY}', srcs=['vY']))
    ops.append(Op('A', 'A3 A3+A6', after=list(sld)))
    return schedule(ops, vregs=[0, 1, 2, 3, 4, 5, 6, 7], sregs=[1, 2, 3, 4, 5, 6, 7],
                    window=window, reserve=reserve,
                    rng=(__import__('random').Random(seed) if seed is not None else None), jitter=jitter)




def generate(layout_path, out_dir, act_bits=6):
    """Write program.cal, rope.img, desc.img and meta.json into out_dir."""
    lay = json.loads(pathlib.Path(layout_path).read_text())
    src, data, meta = build(lay, act_bits)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'program.cal').write_text(src)
    words, corr = rope_table(CTX)
    with open(out / 'rope.img', 'wb') as f:
        f.write(struct.pack('<%dQ' % len(words), *words))
    with open(out / 'desc.img', 'wb') as f:
        for a in sorted(data):
            f.write(struct.pack('<%dQ' % DESC_W, *data[a]))
    meta['rope_corr'] = corr
    meta['model'] = 'ternary'
    meta['act_bits'] = act_bits
    meta['embedding_disk'] = lay['embedding_disk']
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    return out
