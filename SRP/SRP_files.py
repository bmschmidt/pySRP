import numpy as np

class SRP_collection(object):
    """
    An SRP collection is a class for handling a set of SRP objects
    in memory.
    It will include methods for writing to disk, for reading from disk,
    and so forth.

    Unimplemented.
    
    """
    def __init__(self):
        pass

class Vector_file(object):
    """
    A class to manage binary files in the word2vec format.
    I've also adopted this as the binary SRP format.

    One minor problem is that this format doesn't allow for spaces in
    ID names.
    
    Initialized with a filename, a maximum number of rows to read,
    and a maximum number of columns to read.
    """
    def __init__(self,filename,max_rows=float("Inf"),ncol=float("Inf")):
        self.filename = filename
        self.preload_metadata(filename,max_rows)
        self.ncol = ncol
        
    def preload_metadata(self,filename,max_rows):
        """
        A simplified version of the gensim API: https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
        """
        
        counts = None

        self.fin = open(filename)
        header = self.fin.readline()
        self.vocab_size, self.vector_size = map(int, header.split())  # throws for invalid file format

        # Some shuffling to decide whether all the columns are going to be read.
        
        if self.ncol < float("Inf") and self.ncol > self.vector_size:
            import sys
            sys.stderr.write("WARNING: data has only {} columns but call requested top {}"\
                                 .format(self.vector_size,self.ncol))
        if self.ncol == float("Inf") or self.ncol==self.vector_size:
            self.ncol = self.vector_size
            self.slice_and_dice = False
        else:
            self.slice_and_dice = True
        
        self.remaining_words = min([self.vocab_size,max_rows])

    def __iter__(self):
        return self

    def next(self):
        """
        For python2 compatibility
        """
        return self.__next__()

    def __next__(self):
        """
        Again, I'm using a stripped down version of the gensim code.
        https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
        """
        self.remaining_words = self.remaining_words-1
        if self.remaining_words<=-1:
            # Allow breaking out of the loop early.
            raise StopIteration
        fin = self.fin
        # '4' here is the number of bytes in a float.
        binary_len = 4 * self.vector_size
        word = []
        while True:
            ch = fin.read(1)
            if ch == b' ':
                break
            if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                word.append(ch)
            if ch == '\n':
                # Unnecessary.
                word = []
        try:
            word = b''.join(word).decode("utf-8")
        except:
            print(word)
            raise
        
        if self.slice_and_dice:
            # When ncol is less than the resolution of the file size.
            read_length = binary_len - 4*(self.ncol - self.vector_size)
            weights = np.fromstring(fin.read(read_length), dtype='<f4')
            _ = find.read(binary_len - read_length)
        else:
            weights = np.fromstring(fin.read(binary_len), dtype='<f4')            

        
        return (word,weights)
