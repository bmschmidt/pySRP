#### -*- coding: utf-8 -*-
# This code should run under either 2.7 or 3.x

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

class SRP_batch(object):
    def __init__(self, hasher, target_size = 500):
        """
        hasher: an object of class SRP for hashing.
        target_size: the number of **megabytes** to target for the cache.
        
        """
        self.hasher=hasher
        self.proportion = 1500        
        self.matrix_entries = target_size*1024*1024/4
        self.initialize_matrix()
        
    def push(self, docid, countdict):
        (rows,cols) = self.matrix.shape
        i = len(self.rownames)
        if i >= rows:
            raise OverflowError("Matrix is full")        
        for j, word in enumerate(self.colnames):
            try:
                self.matrix[i,j] = countdict[word]
                del countdict[word]                
            except KeyError:
                pass

        if len(countdict) + len(self.colnames) > cols:
            # If it turns out that there's not room, we bail here.
            raise OverflowError("Matrix is full")

        # if not, it's safe to commit to there being a docid element.
        self.rownames.append(docid)

        # Keep incrementing from earlier.
        j = len(self.colnames)
        
        for word,count in countdict.iteritems():
            # These are *new* items
            self.colnames.append(word)
            self.matrix[i,j] = countdict[word]
            j += 1
            
            
    def initialize_matrix(self):
        self.rownames = []
        self.colnames = []        
        nrow = int(np.sqrt(self.matrix_entries/self.proportion))
        ncol = nrow * self.proportion
        self.matrix = np.zeros((nrow,ncol),dtype=np.float32)
        

    def flush(self):
        """
        Transform the current matrix out.
        """
        # Assume the next batch has the same proportions as this one.
        self.proportion = int(len(self.colnames)/len(self.rownames))
        A = self.matrix[0:len(self.rownames),0:len(self.colnames)]
        B = np.zeros((len(self.colnames),self.hasher.dim),np.float32)
        for i,word in enumerate(self.colnames):
            B[i,] = self.hasher.hash_string(word)
        return_val = np.dot(A,B)
        A = ""
        B = ""
        names = self.rownames
        self.initialize_matrix()
        return [
            (names[i], return_val[i])
            for i in range(len(return_val))
            ]
        
class SRP(object):
    """
    A factory to perform random transformations.
    """

    def __init__(self, dim=640, cache=True, cache_limit=500000):
        """
        dim:     The number of dimensions that the transformer
                 should reduce to.

        cache:   Whether to memoize some words. This could cause memory overflows in
                 extremely large document sets that have not had their
                 vocabulary culled down to a few million unique tokens if cache_limit
                 is not set. When it is set, the cache stops marking items once it has
                 cach_limit items in it already: this should catch most common words, but
                 does nothing to ensure that the first n words it catches are useful.

        cache_limit: The maximum cache size.  
        """
        self.dim=dim
        self.cache=cache
        self.cache_limit = cache_limit
        if cache:
            # This is the actual hash.
            self.known_hashes = dict()

    def _cache_size(self):
        return len(self.known_hashes)

    def _expand_hexstring(self, hexstring):
        if py3 and isinstance(hexstring,str):
            h = bytes.fromhex(hexstring)
        elif py2:
            h = hexstring.decode('hex')
        ints = np.fromstring(h, np.uint8)
        value = np.unpackbits(ints).astype(self.dtype)
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
 
        if cache and self._cache_size() < self.cache_limit:
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

    def standardize(self, words, counts, unzip = True):
        full = dict()
        
        for i in range(len(words)):
            """
            Here we retokenize each token. A full text can be tokenized
            at a single pass
            by passing words = [string], counts=[1]
            """
            subCounts = self.tokenize(words[i])
            for (part,partCounts) in subCounts.iteritems():
                addition = counts[i]*partCounts
                try:
                    full[part] += addition
                except KeyError:
                    full[part] = addition
        words = []
        counts = []
        if not unzip:
            return full
        for (k,v) in full.iteritems():
            words.append(k)
            counts.append(v)
        return (words,counts)

    def stable_transform(self,words,counts=None,dim=None,log=True,standardize=True):
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
        # The scores are floats, not ints, because np.dot will
        # use BLAS on a float but not an int.
        scores = np.zeros((len(words), dim), dtype=np.float32)
        for i, word in enumerate(words):
            scores[i] = self.hash_string(word, dim=dim)
        values = np.dot(counts,scores)
        return values
    
    def hash_all_substrings(self, string):
        """
        Breaks a string down into all possible substrings, and then
        returns the projection of the string in the space
        defined by them.

        Possibly useful as a vector-space approximation of string distance.
        """
        counter = Counter()
        
        for i in xrange(len(string)):
            for j in xrange(i + 1, len(string) + 1):
                counter[string[i:j]] += 1

        return self.stable_transform(counter.keys(), counts = counter.values(), log = True, standardize = False)
    
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
