#### -*- coding: utf-8 -*-
# This code should run under either 2.7 or 3.x

from __future__ import absolute_import, division, print_function, unicode_literals
import hashlib
import numpy as np
import regex as re
import sys
import base64

if sys.version_info[0]==3:
    py2,py3 = False,True
    py3 = True
else:
    (py2,py3) = True,False

tokenregex = re.compile(u"(\p{L}+|\p{N}+)")

class SRP(object):
    """
    A factory to perform random transformations.
    """

    def __init__(self,dim=160,cache=True):
        """
        dim:     The number of dimensions that the transformer
                 should reduce to.

        cache:   Whether to memoize This could cause memory overflows in
                 extremely large document sets that have not had their
                 vocabulary culled down to a few million unique tokens.
        """
        self.dim=dim
        self.cache=cache
        if cache:
            # This is the actual hash.
            self.known_hashes = dict()
    
    def _expand_hexstring(self, hexstring):
        if py3 and isinstance(hexstring,str):
            h = bytes.fromhex(hexstring)
        elif py2:
            h = hexstring.decode('hex')
        ints = np.fromstring(h, np.uint8)
        value = np.unpackbits(ints).astype(np.int8)
        value[value == 0] = -1
        return value
    
    def hash_string(self,string,dim=None):
        """
        Gives a hash for a word.
        string:      The string to be hashed.
        dim:         The number of dimensions to hash into.
                     Caching occurs when this dim is the class's
                     number of dimensions.
        """
        # First we check if the cache ought to contain the
        # results; if so, we either return the result, or
        # set a note to enter into the cache when done.
        if dim is None:
            dim=self.dim

        if dim==self.dim and self.cache:
            try:
                return self.known_hashes[string]
            except KeyError:
                cache = True
        else:
            cache = False

        expand = np.ceil(dim / 160).astype('i8')
        full_hash = ""
        for i in range(0,expand):
            seedword = string + "_"*i
            try:
                full_hash += hashlib.sha1(seedword.encode("utf-8")).hexdigest()
            except UnicodeDecodeError:
                full_hash += hashlib.sha1(seedword).hexdigest()

        """
        Do some ugly typecasting
        """

        if py2:
            if isinstance(string,unicode):
                pass# string = string.encode("utf-8")
            else:
                pass
        if py3:
            if isinstance(string,bytes):
                pass# string = string.decode("utf-8")

        value = self._expand_hexstring(full_hash)[:dim]
        
        if cache:
            self.known_hashes[string] = value
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

    def standardize(self,words,counts):
        full = dict()
        for (string,count) in zip(words,counts):
            """
            Here we retokenize each token. A full text can be tokenized
            at a single pass
            by passing words = [string], counts=[1]
            """
            subCounts = self.tokenize(string)
            for (part,partCounts) in subCounts.iteritems():
                try:
                    full[part] += count*partCounts
                except KeyError:
                    full[part] = count*partCounts
        words = []
        counts = []
        for (k,v) in full.iteritems():
            words.append(k)
            counts.append(v)
        return (words,counts)

    def stable_transform(self,words,counts=None,dim=None,log=False,standardize=True):
        """

        """
        if dim is None:
            dim = self.dim
        try:
            if isinstance(words,basestring):
                words = [words]
                counts = [1]
        except NameError:
            # That is, we're in py3
            if isinstance(words,str) or isinstance(words,bytes):
                words = [words]
                counts = [1]            
        if counts is None:
            raise IOError("Counts must be defined when a list of words is passed in.")
        if standardize:
            (words,counts) = self.standardize(words,counts)
        counts = np.array(counts,dtype=np.float32)
        if log:
            # Store as a float because of normalization, etc.
            counts = counts/np.sum(counts)
            counts = np.log(counts*1e05)
            # Anything occurring less than 1 per 100,000 is removed.
            # This lets us avoid negatives, which would screw things up.
            # Once per 100,000 is an arbitrary floor, obv.
            counts.clip(0)
        scores = np.zeros((len(words), dim), dtype=np.int8)
        for i, word in enumerate(words):
            scores[i] = self.hash_string(word, dim=dim)
        values = np.dot(counts,scores)
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
    #print model.stable_transform(["hello", "world"],[1,1])[:6]
