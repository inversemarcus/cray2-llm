"""Shared pieces of the CAL program generators: a small emitter with
register-convention helpers, host-side Cray float constants and the RoPE table."""
import math
import struct


def cf(x):
    """Cray-2 floating representation of a Python float (host-side constant)."""
    if x == 0:
        return 0
    m, e = math.frexp(abs(x))
    return (int(x < 0) << 63) | ((e + 0o40000) << 48) | int(m * (1 << 48))


class Gen:
    def __init__(self, lm, mbox):
        self.lm, self.mbox = lm, mbox
        self.out = []
        self.n = 0

    def e(self, s):
        self.out.append('        ' + s)

    def dbg(self, n):
        """CPU0 stops with EXIT 40+n if bit n of MBOX+5 is set (diagnostics only)."""
        lab = self.uniq('dbg')
        self.e('A1 [%d]' % self.lm['cpu'])
        self.e('JN A1,%s' % lab)
        self.e('S1 (%d)' % (self.mbox + 5))
        self.e('S2 %d' % (1 << n))
        self.e('S1 S1&S2')
        self.e('JZ S1,%s' % lab)
        self.e('EXIT %d' % (40 + n))
        self.lab(lab)

    def lab(self, name):
        self.out.append(name + ':')

    def uniq(self, p):
        self.n += 1
        return '%s_%d' % (p, self.n)

    def call(self, target):
        # 002 with i=k: P <- (A6) and A6 <- return address (HR-2000 3-x)
        self.e('A6 %s' % target)
        self.e('J A6')

    def lm_store_a(self, slot, a):
        self.e('[%d] %s' % (self.lm[slot], a))

    # ------------------------------------------------------------ helpers
    def hsum(self, vr):
        """Horizontal float sum of 64 elements of V<vr> -> S1 (via Local Memory)."""
        self.e('A3 %d' % self.lm['lmt'])
        self.e('[A3] V%d' % vr)
        for w in (32, 16, 8, 4, 2, 1):
            self.e('A4 %d' % w)
            self.e('VL A4')
            self.e('A3 %d' % self.lm['lmt'])
            self.e('V1 [A3]')
            self.e('A3 %d' % (self.lm['lmt'] + w))
            self.e('V2 [A3]')
            self.e('V3 V1+FV2')
            self.e('A3 %d' % self.lm['lmt'])
            self.e('[A3] V3')
        self.e('A4 64')
        self.e('VL A4')
        self.e('S1 [%d]' % self.lm['lmt'])

    def hmax(self, vr, n_label):
        """Horizontal float max of 64 elements of V<vr> -> S1 (via Local Memory)."""
        self.e('A3 %d' % self.lm['lmt'])
        self.e('[A3] V%d' % vr)
        for w in (32, 16, 8, 4, 2, 1):
            self.e('A4 %d' % w)
            self.e('VL A4')
            self.e('A3 %d' % self.lm['lmt'])
            self.e('V1 [A3]')
            self.e('A3 %d' % (self.lm['lmt'] + w))
            self.e('V2 [A3]')
            self.e('V3 V1-FV2')
            self.e('VM V3,P')
            self.e('V4 V1!V2&VM')
            self.e('A3 %d' % self.lm['lmt'])
            self.e('[A3] V4')
        self.e('A4 64')
        self.e('VL A4')
        self.e('S1 [%d]' % self.lm['lmt'])

    def rsqrt(self, s_in, s_out):
        """s_out = 1/sqrt(s_in) (approximation + one documented iteration). Uses S3,S4."""
        self.e('S3 *QS%d' % s_in)
        self.e('S4 S%d*FS3' % s_in)
        self.e('S4 S3*QS4')
        self.e('S%d S3*FS4' % s_out)

    def recip(self, s_in, s_out):
        """s_out = 1/s_in. Uses S3,S4."""
        self.e('S3 /HS%d' % s_in)
        self.e('S4 S3*IS%d' % s_in)
        self.e('S%d S3*FS4' % s_out)

    def set_vl_rem(self, areg, base):
        """VL (and A1) = min(64, base + n - areg).  Clobbers A1, A4."""
        lv = self.uniq('vr')
        self.e('A1 %d' % base)
        self.e('A4 [%d]' % self.lm['n'])
        self.e('A1 A1+A4')
        self.e('A1 A1-%s' % areg)
        self.e('A4 64')
        self.e('A4 A4-A1')
        self.e('JP A4,%s' % lv)
        self.e('A1 64')
        self.lab(lv)
        self.e('VL A1')


def rope_table(npos):
    """YaRN neox cos/sin*mscale per position, emulating ggml's f32 steps."""
    f32 = lambda v: struct.unpack('f', struct.pack('f', v))[0]
    base, fs, ext, beta_fast, beta_slow, n_orig = 1e6, 0.25, 1.0, 32.0, 1.0, 16384

    def corr_dim(nrot):
        return 128 * math.log(n_orig / (nrot * 2 * math.pi)) / (2 * math.log(base))
    low = max(0, math.floor(corr_dim(beta_fast)))
    high = min(127, math.ceil(corr_dim(beta_slow)))
    theta_scale = f32(base ** (-2.0 / 128))
    mscale = f32(1.0 * (1.0 + 0.1 * math.log(1.0 / fs)))
    words = []
    for p in range(npos):
        cs, sn = [], []
        theta = f32(float(p))
        for i in range(64):
            te = theta
            ti = f32(fs * te)
            y = (i - low) / max(0.001, high - low)
            ramp = (1 - min(1.0, max(0.0, y))) * ext
            th = f32(f32(ti * (1 - ramp)) + f32(te * ramp))
            cs.append(cf(f32(f32(math.cos(th)) * mscale)))
            sn.append(cf(f32(f32(math.sin(th)) * mscale)))
            theta = f32(theta * theta_scale)
        words += cs + sn
    return words, dict(low=low, high=high)
