from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import numpy as np

if sys.version_info[0]==3:
    (py2, py3) = False, True
else:
    (py2,py3) = True, False

def textset_to_srp(
        inputfile,
        outputfile,
        dim = 640,
        limit = float("Inf")
        ):
    """
    A convenience wrapper for converting a text corpus to
    an SRP collection as a block.

    inputfile is the collection. The format is the same as the ingest
    format used by bookworm and mallet; that is to say, a single unicode
    file where each line is a document represented as a unique filename, then a tab,
    and then a long string containing the text of the document.
    To coerce into the this format, newlines must be removed.

    outputfile is the SRP file to be created. Recommended suffix is `.bin`.

    dims is the dimensionality of the output SRP.
    
    """
    import SRP
    
    output = Vector_file(outputfile,dims=dim,mode="w")
    hasher = SRP.SRP(dim=dim)

    for i,line in enumerate(open(inputfile,"r")):
        try:
            (id,txt) = line.split("\t",1)
        except:
            print(line)
            raise
        transform = hasher.stable_transform(txt,log=True,standardize=True)
        output.add_row(id,transform)
        if i + 1 >= limit:
            break
        
    output.close()
         
class Vector_file(object):
    """
    A class to manage binary files in the word2vec format.
    I've also adopted this as the binary SRP format.

    One minor problem is that this format doesn't allow for spaces in
    ID names.

    Initialized with a filename, a maximum number of rows to read,
    and a maximum number of columns to read.

    There are three basic ways of operating with one of these.

    1. Treat them as a file object that can read from, line by line, or written
       to, line by line. This is the basic mode, and the only one that supports
       write operations.

    2. Slurp an entire object into memory using the 'as_matrix' method. This returns
       a dict with a matrix and a list of names. This is probably the easiest method
       if the object fits in memory.

    3. Access individual values using dict methods: eg, model['foo'] will return
       the vector representing token 'foo'. The first usage of this method will parse
       the entire file for keys, which may take quite a while; later reads will access an
       in-memory cache of ids to determine where on disk to look, which is significantly faster
       but still slower than an in-memory lookup.
    """
    def __init__(self, filename, dims=float("Inf"), mode="r",max_rows=float("Inf")):
        self.filename = filename
        self.dims = dims
        self.mode = mode
        self.max_rows = max_rows
        if self.mode == "r":
            self._open_for_reading()
        if self.mode == "w":
            self._open_for_writing()
        if self.mode == "a":
            self._open_for_appending()


    def repair_file(self):
        """
        When writing millions of these, sometimes bytes get unaligned: this
        is an imprecise rubric that generally works to fix corrupted data.
        """
        if self.mode != "a":
            raise IOError("Can only repair when in append mode")
        self._preload_metadata()
        # This also sets the pointer to the front.
        self.nrows=0
        self.remaining_words = float("Inf")
        previous_start = self.file.tell()
        while True:
            try:
                _,_ = self.next()
            except StopIteration:
                break
            except UnicodeDecodeError:
                break
            self.nrows+=1
            if self.nrows % 100000==0:
                print("{} read".format(self.nrows))
            previous_start = self.file.tell()
        self.file.truncate(previous_start)
        self._rewrite_header()
        
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
        self.file.seek(0)
        header = "{0:09d} ".format(self.nrows) + "{0:05d}\n".format(self.dims)
        if py2:
            self.file.write(header)
        if py3:
            self.file.write(header.encode("utf-8"))
        # Move pointer to end. Just in case.
        self.file.seek(0,2)
            
    def _open_for_writing(self):
        self.nrows = 0        
        self.file = open(self.filename,"wb")
        self._rewrite_header()
        self.vector_size = self.dims
        
    def _open_for_reading(self):
        self.file = open(self.filename, 'rb')
        self._preload_metadata()

    def to_matrix(self, unit_length = False, clean = False):
        """
        Returns the entire file as a matrix with names (wrapped in a dict).
        
        This can, obviously, overflow memory on a large file.
        """
        labels = []
        matrix = np.zeros((min([self.vocab_size,self.max_rows]),self.dims),"<f4")
        for i,(id,row) in enumerate(self):
            labels.append(id)
            if unit_length:
                row = row/np.linalg.norm(row)
            matrix[i] = row
        return {"names":labels,"matrix":matrix}
            
    def _open_for_appending(self):
        self.file = open(self.filename,"rb+")
        self._preload_metadata()
        self.file.seek(0,2)        
        self.nrows = self.vocab_size
        if self.dims != self.vector_size:
            raise IndexError(
                "The existing files is {} dimensions: unable to append"
                " with {} dimensions as requested".format(
                    self.vector_size,self.dims))

    def _recover_from_corruption(self):
        starting = self.pos
        self.debug_mode = True
        for i in range(2000000):
            self.file.seek(starting + i)
            try:
                gah = self.next()
                safe_pos = self.pos
                for n in range(10):
                    # Make sure things are relatively straightforward from here on out.
                    _ = self.next()
                self.file.seek(safe_pos)
                del self.debug_mode
                return True
            except StopIteration:
                pass
        print("Encountered corrupted data with {} words left and unable to recover at all".format(self.remaining_words))
        raise StopIteration            
        
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
        if py2:
            try:
                print(identifier)
                self.file.write(identifier)
            except UnicodeEncodeError:
                # Don't do this at first to allow writing of already encoded unicode
                self.file.write(identifier.encode("utf-8"))
            self.file.write(b" ")
            self.file.write(array.tobytes())
            self.file.write(b"\n")
        if py3:
            self.file.write(identifier.encode("utf-8") + b" ")
            self.file.write(array.tobytes())
            self.file.write(b"\n")
        self.nrows += 1
        
    def close(self):
        """
        Close the file. It's extremely important to call this method in write modes:
        not just that the last few files will be missing.        
        If it isn't, the header will have out-of-date information and files won't be read.
        """
        if not "r" in self.mode:
            self._rewrite_header()

        self.file.close()
            
    def _preload_metadata(self):
        """
        A simplified version of the gensim API.

        From https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py        
        """

        counts = None
        self.file.seek(0)
        header = self.file.readline()
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

    def __getitem__(self):
        pass

    
    
    def _read_row_name(self):
        buffer = []
        while True:
            ch = self.file.read(1)
            if not ch:
                print("Ran out of data with {} words left".format(self.remaining_words))
                return
            if ch == b' ':
                break
            if ch != b'\n':
                # ignore newlines in front of words (some binary files have em)
                buffer.append(ch)
        try:
            word = b''.join(buffer).decode("utf-8")
        except TypeError:
            # We're in python 3
            word = ''.join(word)
        except UnicodeDecodeError:
            if not hasattr(self,"debug_mode"):
                self._recover_from_corruption()
            else:
                raise
        except:
            print("Couldn't export:")
            print(word)
            raise
        return word

    def _build_offset_lookup(self, force = False):
        if hasattr(self, "_offset_lookup") and not force:
            return
        self._offset_lookup = {}
        self._preload_metadata()
        while len(self._offset_lookup) < self.vocab_size:
            label = self._read_row_name()
            self._offset_lookup[label] = self.file.tell()
            # Skip to the next name without reading.
            self.file.seek(4*self.vector_size, 1)

    def __getitem__(self, label):
        self._build_offset_lookup()
        needed_position = self._offset_lookup[label]
        self.file.seek(needed_position)
        return self._read_binary_row()
        
    def _read_binary_row(self):
        binary_len = 4 * self.vector_size
        self.pos = self.file.tell()
        if self.slice_and_dice:
            # When dims is less than the resolution of the file size.
            read_length = 4*self.dims
            weights = np.frombuffer(self.file.read(read_length), dtype='<f4')
            # Catch up the pointer.
            _ = self.file.read(4*self.vector_size - read_length)
        else:
            try:
                weights = np.frombuffer(self.file.read(binary_len), dtype='<f4')            
            except ValueError:
                print("Can't parse data with {} words left".format(self.remaining_words))
                raise StopIteration
            if len(weights) != self.vector_size:
                print("Ran out of data with {} words left".format(self.remaining_words))
                raise StopIteration                
        return weights
    
    def __iter__(self):
        """
        Again, I'm starting with a version of the gensim code.
        https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
        """

        # Always preload the metadata so you can iterate multiple times.
        self._preload_metadata()
        while True:
            self.remaining_words = self.remaining_words-1
            if self.remaining_words<=-1:
                # Allow breaking out of the loop early.
                return
            word = self._read_row_name()
            weights = self._read_binary_row()
            # '4' here is the number of bytes in a float.
            yield (word,weights)
