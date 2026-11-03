"""Pure-Python byte-level BPE tokenizer for the Qwen2/Qwen3 vocabulary stored in a
GGUF file (tokenizer.ggml.model = gpt2, pre = qwen2).

This is host-side text <-> token-ID conversion only (the Foreground
Processor's job); it performs no model arithmetic.  It reproduces
llama.cpp's llama_tokenize(..., parse_special=true) and
llama_detokenize(..., remove_special=false, unparse_special=false), and is
checked against it by tests/test_tokenizer.py.
"""
import json
import os
import unicodedata

from .gguf import GGUF


def _bytes_to_unicode():
    # byte-to-character table from OpenAI's GPT-2 encoder.py (MIT); see THIRD_PARTY.md
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_B2U = _bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}


def _is_letter(c):
    return unicodedata.category(c)[0] == 'L'


def _is_number(c):
    return unicodedata.category(c)[0] == 'N'


def _is_space(c):
    return c.isspace()


def pretokenize(text):
    """Split text like the qwen2 pre-tokenizer regex:
    (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}
    | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # 1. contractions
        if c == "'" and i + 1 < n:
            low = text[i + 1:i + 3].lower()
            if low[:2] in ('re', 've', 'll'):
                out.append(text[i:i + 3]); i += 3; continue
            if low[:1] in ('s', 't', 'm', 'd'):
                out.append(text[i:i + 2]); i += 2; continue
        # 2. optional non-letter/number/newline char, then letters
        j = i
        if not (c in '\r\n' or _is_letter(c) or _is_number(c)) and i + 1 < n and _is_letter(text[i + 1]):
            j = i + 1
        if _is_letter(text[j]):
            k = j
            while k < n and _is_letter(text[k]):
                k += 1
            out.append(text[i:k]); i = k; continue
        # 3. single number
        if _is_number(c):
            out.append(c); i += 1; continue
        # 4. optional space, punctuation run, trailing newlines
        j = i + 1 if (c == ' ' and i + 1 < n) else i
        def punct(ch):
            return not (_is_space(ch) or _is_letter(ch) or _is_number(ch))
        if j < n and punct(text[j]):
            k = j
            while k < n and punct(text[k]):
                k += 1
            while k < n and text[k] in '\r\n':
                k += 1
            out.append(text[i:k]); i = k; continue
        # whitespace alternatives
        if _is_space(c):
            j = i
            while j < n and _is_space(text[j]):
                j += 1
            # 5. \s*[\r\n]+
            last = -1
            for k in range(j - 1, i - 1, -1):
                if text[k] in '\r\n':
                    last = k
                    break
            if last >= 0:
                out.append(text[i:last + 1]); i = last + 1; continue
            # 6. \s+(?!\S)
            if j == n:
                out.append(text[i:j]); i = j; continue
            if j - 1 > i:
                out.append(text[i:j - 1]); i = j - 1; continue
            # 7. \s+
            out.append(text[i:j]); i = j; continue
        out.append(c)
        i += 1
    return out


class Tokenizer:
    def __init__(self, gguf_path=None, cache=None):
        """Load from a vocabulary JSON (cache) if it exists, else read the GGUF
        file and write the cache (default ~/.cache/cray2lm)."""
        if cache is None:
            d0 = os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'cray2lm')
            os.makedirs(d0, exist_ok=True)
            st = os.stat(gguf_path)
            cache = os.path.join(d0, '%s.%d.vocab.json' % (os.path.basename(gguf_path), st.st_size))
        if os.path.exists(cache):
            d = json.load(open(cache, encoding='utf-8'))
        else:
            md = GGUF(gguf_path).metadata
            assert md['tokenizer.ggml.model'] == 'gpt2' and md['tokenizer.ggml.pre'] == 'qwen2'
            d = dict(tokens=md['tokenizer.ggml.tokens'], types=md['tokenizer.ggml.token_type'],
                     merges=md['tokenizer.ggml.merges'], eos=md.get('tokenizer.ggml.eos_token_id'))
            with open(cache, 'w', encoding='utf-8') as f:
                json.dump(d, f)
        self.tokens, self.types = d['tokens'], d['types']
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {tuple(m.split(' ', 1)): r for r, m in enumerate(d['merges'])}
        # control (3) and user-defined (4) tokens are matched literally in the input
        self.special = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (3, 4)), key=len, reverse=True)
        self.eog = {i for i, t in enumerate(self.tokens)
                    if t in ('<|endoftext|>', '<|im_end|>', '<|fim_pad|>', '<|repo_name|>', '<|file_sep|>')}
        if d.get('eos') is not None:
            self.eog.add(d['eos'])
        self._cache = {}

    def _bpe(self, word):
        if word in self._cache:
            return self._cache[word]
        parts = [_B2U[b] for b in word.encode('utf-8')]
        while len(parts) > 1:
            best, bi = None, -1
            for k in range(len(parts) - 1):
                r = self.ranks.get((parts[k], parts[k + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, k
            if best is None:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        ids = [self.vocab[p] for p in parts]
        self._cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            # earliest special token occurrence (longest at that position)
            best = None
            for s in self.special:
                k = text.find(s, i)
                if k >= 0 and (best is None or k < best[0] or (k == best[0] and len(s) > len(best[1]))):
                    best = (k, s)
            end = best[0] if best else len(text)
            for w in pretokenize(text[i:end]):
                ids += self._bpe(w)
            if not best:
                break
            ids.append(self.vocab[best[1]])
            i = best[0] + len(best[1])
        return ids

    def decode_bytes(self, ids):
        out = bytearray()
        for t in ids:
            ty = self.types[t]
            if ty == 3:            # control tokens render as nothing
                continue
            if ty == 4:            # user-defined tokens render as their text
                out += self.tokens[t].encode('utf-8')
                continue
            out += bytes(_U2B[ch] for ch in self.tokens[t])
        return bytes(out)

    def decode(self, ids):
        return self.decode_bytes(ids).decode('utf-8', errors='replace')

    def is_eog(self, t):
        return t in self.eog
