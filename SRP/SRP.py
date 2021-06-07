#### -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import hashlib
import numpy as np
import regex
import sys
import base64
from collections import Counter

tokenregex = regex.compile(u"\w+")

class EmptyTextError(ZeroDivisionError):
    pass

class SRP(object):
    """
    A factory to perform random transformations.
    """

    def __init__(self, dim=640, cache=True, cache_limit=15e05, log = True):
        """
        dim:     The number of dimensions that the transformer
                 should reduce to.

        cache:   Whether to memoize some words. This could cause memory overflows in
                 extremely large document sets that have not had their
                 vocabulary culled down to a few million unique tokens if cache_limit
                 is not set. When it is set, the cache stops marking items once it has
                 cach_limit items in it already: this should catch most common words, but
                 does nothing to ensure that the first n words it catches are useful.

        cache_limit: The maximum cache size. Once the cache hits this size, it is deleted
                 and starts over: so if this size is too small (less than 100,000, say)
                 performance will actually be worse. Recommended is over a million.

        log:     Use a log transform? Usually useful, but in cases where function words matter
                 more, it may not be.
        """
        self.dim=dim
        self.cache = cache
        self.cache_limit = cache_limit
        self.dtype = np.float32
        self.log = log

        if cache:
            # This is the actual hash.
            self._hash_dict = dict()
            self._last_hash_dict = dict()

        self.build_lookup_table()

    def _cache_size(self):
        return len(self._hash_dict)

    def build_lookup_table(self):
        """
        A table mapping four digit hexadecimal numbers
        to 16-bit SRP vectors in 1, -1 space.
        """
        lookup = np.zeros((2**16, 16), np.float32)
        for i in range(2**16):
#            str_rep = f"{i:04x}"
#            h = bytes.fromhex(str_rep)
#            ints = np.frombuffer(h, np.uint8)
            bits = np.array([i], np.uint16)
            value = np.unpackbits(bits.view(np.uint8)).astype(np.int8)
            lookup[i] = value * 2 - 1
        self.lookup_table = lookup

    def string_to_binary(self, string):
        expand = np.ceil(self.dim / 160).astype('i8')
        full_hash = b""
        for i in range(0, expand):
            seedword = string + "_" * i
            try:
                full_hash += hashlib.sha1(seedword.encode("utf-8")).digest()
            except UnicodeDecodeError:
                full_hash += hashlib.sha1(seedword).digest()
        as_keys = np.frombuffer(full_hash, np.uint16)
        return self.lookup_table[as_keys].flatten()[:self.dim]

    def old_string_to_binary(self, string):
        expand = np.ceil(self.dim / 160).astype('i8')
        full_hash = ""
        hash = np.float32()
        for i in range(0, expand):
            seedword = string + "_" * i
            try:
                full_hash += hashlib.sha1(seedword.encode("utf-8")).hexdigest()
            except UnicodeDecodeError:
                full_hash += hashlib.sha1(seedword).hexdigest()
        return self._expand_hexstring(full_hash)

    def hash_string(self, string, cache=None):

        """
        Gives a hash for a word.
        string:      The string to be hashed.
        cache:       Whether to cache the result. "None"
                     uses the default for object;
                     False turns it off.
        """
        # First we check if the cache ought to contain the
        # results; if so, we either return the result, or
        # set a note to enter into the cache when done.


        if self.cache:
            try:
                return self._hash_dict[string]
            except KeyError:
                try:
                    # Check the previous cache.
                    self._hash_dict[string] = self._last_hash_dict[string]
                    return self._hash_dict[string]
                except KeyError:
                    if cache is None:
                        cache = True
        else:
            cache = False

        value = self.string_to_binary(string)

        if cache and self._cache_size() >= self.cache_limit:
            # Clear the cache; maybe things have changed.
            self._hash_dict = self._last_hash_dict
            self._hash_dict = {}

        if cache and self._cache_size() < self.cache_limit:
            self._hash_dict[string] = value

        return value

    def tokenlist(self, string, regex = tokenregex, lower = True):
        if isinstance(string, bytes):
            string = string.decode("utf-8")
        return regex.findall(string)

    def tokenize(self, string, regex=tokenregex, lower = True):
        parts = self.tokenlist(string, regex, lower)
        count = dict()
        for part in parts:
            if lower:
                part = part.lower()
            try:
                count[part] += 1
            except KeyError:
                count[part] = 1
        return count

    def standardize(self, words, counts, unzip = True):
        full = dict()

        for i in range(len(words)):
            """
            Here we retokenize each token. A full text can be tokenized
            at a single pass
            by passing words = [string], counts=[1]
            """
            subCounts = self.tokenize(words[i])
            for (part,partCounts) in subCounts.items():
                part = regex.sub(u'\d',"#",part)
                addition = counts[i]*partCounts
                try:
                    full[part] += addition
                except KeyError:
                    full[part] = addition
        words = []
        counts = np.zeros(len(full),"<f4")
        if not unzip:
            return full
        for i,(k,v) in enumerate(full.items()):
            words.append(k)
            counts[i] = v
        return (words,counts)

    def _str_to_wordcounts(self, words):
        # Note that wordcounts are floats though ints would
        # be more sensical. This is to force the use of BLAS
        # for matrix multiplication, which seems to be faster.

        if isinstance(words, str) or isinstance(words, bytes):
            words = [words]
            counts = np.array([1], np.float32)

        return (words,counts)

    def _log_transform(self,counts,thresh = 1e05):
        # Take a ratio of the full text.
        total = np.sum(counts)
        if total == 0:
            raise EmptyTextError("Can't normalize, zero words in text.")
        counts = counts/total
        counts = np.log(counts*thresh)
        # Anything occurring less than 1 per 100,000 is removed.
        # This lets us avoid negatives, which would screw things up.
        # Once per 100,000 is an arbitrary floor, obv.
        counts.clip(0)
        return counts

    def stable_transform(self, words, counts=None, log=None, standardize=True,
                         unit_length = True
        ):
        """
        words: either a list of words, or a single string.
        counts: the number of occurrences for each word in 'words'. This can be "none",
                in which 'words' is treated as a string.
        log: Apply a log-transform to avoid common words dominating the signature?
        standardize: Apply standard minimal tokenization rules?
        unit_length: normalize each vector to unit length to speed up
                     subsequent calculations on cosine distance?
        """

        if log is None:
            log = self.log
        if counts is None:
            (words,counts) = self._str_to_wordcounts(words)
        if standardize:
            (words,counts) = self.standardize(words,counts)
        if log:
            counts = self._log_transform(counts)
        scores = np.zeros((len(words), self.dim), dtype = np.float32)
        for i, word in enumerate(words):
            scores[i] = self.hash_string(word)
        try:
#            values = np.average(scores, weights=counts, axis=0)
            values = counts @ scores
#            if unit_length == False:
#                values = values * sum(counts)
        except ZeroDivisionError:
            if sum(counts) == 0:
                raise EmptyTextError
            else:
                raise
        if unit_length:
            values = values/np.linalg.norm(values)
        return values


    def to_base64(self,vector):
        """
        Converts a vector to a base64, little-endian, 4-byte representation
        in base 64.
        """
        string = np.array(vector,'<f4')
        return base64.b64encode(string)

if __name__=="__main__":
    model = SRP(320)
    print(model.stable_transform("hello world")[:6])
    model = SRP(320)
    print(model.stable_transform(u"Güten Tag")[:6])
    model = SRP(320)
    print(model.stable_transform(u"Güten Tag".encode("utf-8").decode("utf-8"))[:6])
    model = SRP(320)
    print(model.stable_transform(u"Güten Tag".encode("utf-8"))[:6])
