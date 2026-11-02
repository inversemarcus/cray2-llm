"""Compare cray2lm.tokenizer with llama.cpp's tokenizer.

    python3 tests/test_tokenizer.py MODEL.gguf REF_TOKENIZER_CMD... [--texts file.jsonl ...]

REF_TOKENIZER_CMD is tools/llama_tokenizer.cpp built against llama.cpp
(PrismML-Eng/llama.cpp at a153697, which supports these tensor types), e.g.

    c++ -std=c++17 -O2 -I$LLAMA/include -I$LLAMA/ggml/include tools/llama_tokenizer.cpp \\
        $LLAMA/build/src/libllama.a $LLAMA/build/ggml/src/libggml*.a -fopenmp -o llama_tokenizer
    python3 tests/test_tokenizer.py MODEL.gguf ./llama_tokenizer MODEL.gguf

Every prompt string found in the given JSONL files (keys text/prompt/context/question) is checked, plus
built-in edge cases and random strings.  Exits non-zero on any mismatch.
"""
import json
import os
import random
import struct
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cray2lm.tokenizer import Tokenizer  # noqa: E402


class RefTokenizer:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _rpc(self, op, n=0, data=b''):
        self.p.stdin.write(struct.pack('<cQ', op, n) + data)
        self.p.stdin.flush()
        status, size = struct.unpack('<cQ', self.p.stdout.read(9))
        res = self.p.stdout.read(size)
        if status != b'O':
            raise RuntimeError(res.decode())
        return res

    def encode(self, text):
        d = text.encode()
        r = self._rpc(b'E', len(d), d)
        return list(struct.unpack('<%di' % (len(r) // 4), r))

    def decode_bytes(self, ids):
        return self._rpc(b'D', len(ids), struct.pack('<%di' % len(ids), *ids))


EDGE = ["Hello world", "  leading", "trailing  ", "a\n\n\nb", "x \n y", "tabs\t\tand  spaces   ",
        "it's they'RE we'll I'd you've", "12345 3.14159 1,000,000", "naïve café résumé", "日本語のテキスト",
        "Привет мир", "emoji 🎉🚀 test",
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "C++ & Python -> ok?!...", "\r\n\r\nwindows\r\nlines", "   \n   ", "a  b   c    d",
        "foo@bar.com https://example.org/x?y=1"]


def main():
    args = sys.argv[1:]
    texts_files = []
    if '--texts' in args:
        k = args.index('--texts')
        texts_files = args[k + 1:]
        args = args[:k]
    model, ref_cmd = args[0], args[1:]
    texts = list(EDGE)
    for fn in texts_files:
        for line in open(fn, encoding='utf-8'):
            d = json.loads(line)
            for key in ('text', 'prompt', 'context', 'question'):
                if isinstance(d.get(key), str):
                    texts.append(d[key])
    rnd = random.Random(1)
    alphabet = "aAzZ09 \n\t.,;:!?'\"()[]{}-_/\\@#$%^&*+=<>|~`éßΩж中🙂"
    texts += [''.join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 40))) for _ in range(400)]
    mine, ref = Tokenizer(model), RefTokenizer(ref_cmd)
    bad = dbad = 0
    for s in texts:
        a, b = mine.encode(s), ref.encode(s)
        if a != b:
            bad += 1
            if bad <= 5:
                print('ENCODE', repr(s[:70]), a[:12], b[:12])
        if mine.decode_bytes(b) != ref.decode_bytes(b):
            dbad += 1
            if dbad <= 5:
                print('DECODE', repr(s[:70]))
    print('%d strings: %d encode mismatches, %d decode mismatches' % (len(texts), bad, dbad))
    sys.exit(1 if bad or dbad else 0)


if __name__ == '__main__':
    main()
