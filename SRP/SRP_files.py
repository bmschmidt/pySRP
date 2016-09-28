import numpy as np

"""
Maybe not happening after all.
class SRP_collection(object):
    " ""
    An SRP collection is a class for handling a set of SRP objects
    in memory.
    It will include methods for writing to disk, for reading from disk,
    and so forth.

    Unimplemented.
    " ""
    def __init__(self):
        pass
"""

    
class Vector_file(object):
    """
    A class to manage binary files in the word2vec format.
    I've also adopted this as the binary SRP format.

    One minor problem is that this format doesn't allow for spaces in
    ID names.

    Initialized with a filename, a maximum number of rows to read,
    and a maximum number of columns to read.
    """
    def __init__(self, filename, dims=float("Inf"), mode="r",max_rows=float("Inf")):
        self.filename = filename
        self.dims = dims
        self.mode = mode
        self.max_rows = max_rows
        if self.mode == "r":
            self._preload_metadata()
        if self.mode == "w":
            self._open_for_writing()
        if self.mode == "a":
            self._open_for_appending()

    def concatenate_file(self,filename):
        """
        Adds all record in a different file to the end of this one.
        Useful if creation of files has been parallelized or distributed.

        In theory, it should be possible to add a large file to a smaller one.
        """
        
        new_file = Vector_file(filename, dims = self.dims, mode="r")
        for (id, array) in new_file:
            self.add_row(id, array)
        
    def _rewrite_header(self):
        """
        Overwrites the first line of a binary file (where file length and number of columns
        are stored.)
        """
        self.fout.seek(0)
        #print self.dims
        header = "{0:09d} ".format(self.nrows) + "{0:05d}\n".format(self.dims)
        self.fout.write(header)
        # Move pointer to end. Just in case.
        self.fout.seek(0,2)
            
    def _open_for_writing(self):
        self.nrows = 0        
        self.fout = open(self.filename,"w")
        self._rewrite_header()
        self.vector_size = self.dims
        
    def _open_for_appending(self):
        self.fout = open(self.filename,"a")
        self._preload_metadata()
        self.nrows = self.vocab_size
        if self.dims != self.vector_size:
            raise IndexError(
                "The existing files is {} dimensions: unable to append"
                " with {} dimensions as requested".format(
                    self.vector_size,self.dims))

    def add_row(self, identifier, array):
        """
        Add a new document/word/whatever to the matrix.
        """
        if type(array) != np.ndarray:
            raise TypeError("Must pass a numpy ndarray as array")
        if array.dtype != np.dtype('<f4'):
            raise TypeError("Numpy array must be of type '<f4'")
        if len(array) != self.dims:
            raise IndexError("The existing files is {} dimensions: unable to append with {} dimensions as requested".format(self.vector_size,self.dims))

        self.fout.write("{} {}\n".format(identifier,array.tobytes()))
        self.nrows += 1
        
    def close(self):
        """
        Close the file. It's extremely important to call this method; if it isn't, the header will have out-of-date information.
        """
        if not "r" in self.mode:
            self._rewrite_header()
            self.fout.close()
        else:
            self.fin.close()
            
    def _preload_metadata(self):
        """
        A simplified version of the gensim API.

        From https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py

        
        """

        counts = None


        self.fin = open(self.filename)
        header = self.fin.readline()
        self.vocab_size, self.vector_size = map(int, header.split())  # throws for invalid file format

        # Some shuffling to decide whether all the columns are going to be read.

        if self.dims < float("Inf") and self.dims > self.vector_size:
            import sys
            sys.stderr.write(
                "WARNING: data has only {} columns but call requested top {}".format(
                    self.vector_size,self.dims))
        if self.dims == float("Inf") or self.dims==self.vector_size:
            self.dims = self.vector_size
            self.slice_and_dice = False
        else:
            self.slice_and_dice = True
        
        self.remaining_words = min([self.vocab_size,self.max_rows])

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
            # When dims is less than the resolution of the file size.
            read_length = binary_len - 4*(self.dims - self.vector_size)
            weights = np.fromstring(fin.read(read_length), dtype='<f4')
            _ = find.read(binary_len - read_length)
        else:
            try:
                weights = np.fromstring(fin.read(binary_len), dtype='<f4')            
            except ValueError:
                print "Ran out of data with {} words left".format(self.remaining_words)
                raise StopIteration
        return (word,weights)
