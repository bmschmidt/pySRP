# -*- coding: utf-8 -*-
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

        if dim > 160:
            # For longer than 160, we keep recursing and adding underscore characters to the end.
            # Caching is disabled b/c the subsets should not be useful.
            start = self.hash_string(string,dim=160)
            try:
                nextbit = string + "_"
            except UnicodeDecodeError:
                # Maybe someone has passed in already-encoded unicode.
                nextbit = string.decode("utf-8") + "_"
            remainder = self.hash_string(nextbit,dim = dim-160)
                
            value = np.concatenate((start,remainder))

        if dim < 160:
            start = self.hash_string(string,dim=160)
            value = start[:dim]

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

        if dim==160:
            # This is where it actually happens.
            try:
                hashed = hashlib.sha1(string.encode("utf-8")).hexdigest()
            except UnicodeDecodeError:
                hashed = hashlib.sha1(string).hexdigest()
            binary = str(bin(int(hashed, 16))[2:].zfill(160))
            value = np.array([-1 if i=="0" else 1 for i in binary],np.float)
        if cache:
            self.known_hashes[string] = value
        return value

    def tokenize(self,string,regex=u"(\p{L}+|\p{N}+)"):
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
        parts = re.findall(regex,string)
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
            for (part,partCounts) in subCounts.items():
                try:
                    full[part] += count*partCounts
                except KeyError:
                    full[part] = count*partCounts
        newvals = [(k,v) for k,v in full.items()]
        return zip(*newvals)

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
        counts = np.array(counts,dtype=np.float)
        if log:
            # Store as a float because of normalization, etc.
            counts = counts/np.sum(counts)
            counts = np.log(counts*1e05)
            # Anything occurring less than 1 per 100,000 is removed.
            # This lets us avoid negatives, which would screw things up.
            # Once per 100,000 is an arbitrary floor, obv.
            counts.clip(0)
        wordhashes = [self.hash_string(word,dim=dim) for word in words]
        scores = np.array(wordhashes,np.float)
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

