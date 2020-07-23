# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
from multiprocessing import Pool
import os

import torch

from fairseq.tokenizer import tokenize_line
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils
from fairseq import distributed_utils


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""
    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>', sep='<sep>'):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.sep_word = sep
        self.symbols = []
        self.count = []
        self.indices = {}
        # dictionary indexing starts at 1 for consistency with Lua
        self.add_symbol('<Lua heritage>')
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.sep_index = self.add_symbol(sep)
        self.nspecial = len(self.symbols)
        print('N special symbols: ', self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False, tgt_dict=None, escape_pad=False):
        """Helper for converting a tensor of token indices to a string.
        fairseq/data/dictionary.py:76
        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, tgt_dict=tgt_dict, escape_pad=escape_pad) for t in tensor)

        def token_string(i,idx,sep_id=None):
            if i == self.unk():
                return self.unk_string(escape_unk)
            elif tgt_dict is not None:
                word=tgt_dict[i] if ((sep_id > 0) and (idx > sep_id)) else self[i]
                return word
            elif escape_pad and i==self.pad():
                return ''
            else:
                return self[i]
        
        token_seq = [int(x) for x in tensor]
        sep_id = token_seq.index(self.sep_index) if (self.sep_index in token_seq) else -1
        sent = ' '.join(token_string(i,idx,sep_id=sep_id) \
            for idx,i in enumerate(tensor) if i != self.eos())
        return data_utils.process_bpe_symbol(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[:self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]

        c = Counter(dict(zip(self.symbols[self.nspecial:], self.count[self.nspecial:])))
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        threshold_nwords = len(new_symbols)
        if padding_factor > 1:
            i = 0
            while threshold_nwords % padding_factor != 0:
                symbol = 'madeupword{:04d}'.format(i)
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(0)
                i += 1
                threshold_nwords += 1

        assert len(new_symbols) % padding_factor == 0
        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index
    
    def sep(self):
        """Helper to get index of unk symbol"""
        return self.sep_index

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        return cls.load(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = cls()
        lines = f.readlines()
        indices_start_line = d._load_meta(lines)
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx + 1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print('{} {}'.format(k, v), file=f)

    def _save_nspecial(self, f, kv_iterator):
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save_nspecial(fd)
        for k, v in kv_iterator:
            print('{} {}'.format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(f, zip(ex_keys + self.symbols[self.nspecial:], ex_vals + self.count[self.nspecial:]))

    def save_nspecial(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save_nspecial(f, zip(ex_keys + self.symbols[:self.nspecial], ex_vals + self.count[:self.nspecial]))

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(self, line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                    consumer=None, append_eos=True, reverse_order=False, tgt_dict=None):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1  if append_eos else nwords)
        if tgt_dict is not None:
            sep_idx = words.index('<sep>') if '<sep>' in words else -1

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            elif tgt_dict is None:
                idx = self.index(word)                    
            else:
                idx = self.index(word) if (i < sep_idx or sep_idx < 0) else tgt_dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in counter.items():
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    Dictionary._add_file_to_dictionary_single_worker,
                    (filename, tokenize, dict.eos_word, worker_id, num_workers)
                ))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(Dictionary._add_file_to_dictionary_single_worker(filename, tokenize, dict.eos_word))

class TruncatedDictionary(object):

    def __init__(self, wrapped_dict, length):
        self.__class__ = type(
            wrapped_dict.__class__.__name__,
            (self.__class__, wrapped_dict.__class__),
            {}
        )
        self.__dict__ = wrapped_dict.__dict__
        self.wrapped_dict = wrapped_dict
        self.length = min(len(self.wrapped_dict), length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.length:
            return self.wrapped_dict[i]
        return self.wrapped_dict.unk()
