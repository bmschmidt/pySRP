#### -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import hashlib
import numpy as np
import regex
import sys
import base64
from collections import Counter


if sys.version_info[0]==3:
    py2,py3 = False,True
    py3 = True
else:
    (py2,py3) = True,False

tokenregex = regex.compile(u"\w+")

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
        self.cache=cache
        self.cache_limit = cache_limit
        self.dtype = np.float32
        self._recurse_dict = {}
        self._scaled_recurse_dict = {}
        self.log = log
        
        if cache:
            # This is the actual hash.
            self._hash_dict = dict()

    def _cache_size(self):
        return len(self._hash_dict)

    def _expand_hexstring(self, hexstring):
        
        if py3 and isinstance(hexstring,str):
            h = bytes.fromhex(hexstring)
        elif py2:
            h = hexstring.decode('hex')
        ints = np.frombuffer(h, np.uint8)
        value = np.unpackbits(ints).astype(np.int8)
        return (value*2-1)[:self.dim]
        
    def string_to_binary(self, string):
        expand = np.ceil(self.dim / 160).astype('i8')
        full_hash = ""
        for i in range(0,expand):
            seedword = string + "_"*i
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
                if cache is None:
                    cache = True
        else:
            cache = False

        value = self.string_to_binary(string)
            
        if cache and self._cache_size() >= self.cache_limit:
            # Clear the cache; maybe things have changed.
            self._hash_dict = {}
        
        if cache and self._cache_size() < self.cache_limit:
            self._hash_dict[string] = value
        
        return value

    def tokenize(self,string,regex=tokenregex):
        if py3 and isinstance(string,bytes):
            string = string.decode("utf-8")
        if py2 and not isinstance(string,unicode):
            try:
                string = unicode(string)
            except UnicodeDecodeError:
                try:
                    string = string.decode("utf-8")
                except UnicodeDecodeError:
                    sys.stderr.write("Encountered non-unicode string" + "\n")
                    string = string.decode("utf-8","ignore")
        count = dict()
        parts = regex.findall(string)
        for part in parts:
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

        try:        
            if isinstance(words,basestring):
                words = [words]
                counts = np.array([1],np.float32)

        except NameError:
            # That is, we're in py3
            if isinstance(words,str) or isinstance(words,bytes):
                words = [words]
                counts = np.array([1],np.float32)
        
        return (words,counts)

    def _log_transform(self,counts,thresh = 1e05):
        # Take a ratio of the full text.
        counts = counts/np.sum(counts)
        counts = np.log(counts*thresh)
        # Anything occurring less than 1 per 100,000 is removed.
        # This lets us avoid negatives, which would screw things up.
        # Once per 100,000 is an arbitrary floor, obv.
        counts.clip(0)
        return counts

    def stable_transform(self, words, counts=None, log=None,standardize=True):
        """

        counts: the number of occurrences for each word in 'words'. This can be "none",
                in which 'words' is treated as a string.

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

        values = np.dot(counts,scores)

        return values
            
    def hash_all_substrings(self, string,depth=0):
        """
        Breaks a string down into all possible substrings, and then
        returns the projection of the string in the space
        defined by them.

        Possibly useful as a vector-space approximation of string distance.
        """
        counter = Counter()
        if string in self._recurse_dict:
            return self._recurse_dict[string]
        elif len(self._recurse_dict) > self.cache_limit * 2:
            self._recurse_dict = {}
        # The subhash is the hash of this string,
        # plus the hash of its two longest substrings,
        # minus their overlap (which is counted twice).
        # So hash(foobar) =
        # SRP(foobar) + hash(oobar) +
        # hash(fooba) - hash(ooba).

        # This recursive formulation should work fine
        # with caching.


        # Start with the current string.
        output = self.hash_string(string, cache=False)

        # The cases where recursion ends.
        if len(string) == 1:
            self._recurse_dict[string] = output
            return output
        if len(string) == 2:
            self._recurse_dict[string] = output            
            return output + self.hash_all_substrings(string[0],depth+1) + \
                self.hash_all_substrings(string[1],depth+1)

        # Recurse down the parts.
        start = string[:-1]
        end = string[1:]
        middle = string[1:-1]

        output = output + self.hash_all_substrings(start,depth+1) +\
                 self.hash_all_substrings(end,depth+1) -\
                 self.hash_all_substrings(middle,depth+1)
        
        self._recurse_dict[string] = output
        return output        
    
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
    #print model.stable_transform(["hello", "world"],[1,1])[:6]
