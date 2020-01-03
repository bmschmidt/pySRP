from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import numpy as np
import warnings
import os
import collections
import random
import regex as re
import re as original_re


regex_type = type(re.compile("."))
original_regex_type = type(original_re.compile("."))

if sys.version_info[0] == 3:
    (py2, py3) = False, True
    from collections.abc import MutableSequence
else:
    (py2, py3) = True, False
    from collections import MutableSequence
    # Kludgy
    FileNotFoundError = IOError

def textset_yielder(inputfile):
    for line in open(inputfile, "r"):
        try:
            (id, txt) = line.split("\t", 1)
        except:
            print(line)
            raise
        yield (id, txt)

def directory_yielder(inputfile):
    for filename in os.listdir(inputfile):
        if filename.endswith(".txt"):
            id = filename[:-4]
            txt = "\n".join(open(os.path.join(inputfile, filename)).readlines())
            yield (id, txt)
            
def textset_to_srp(
        inputfile,
        outputfile,
        dim=640,
        limit=float("Inf"),
        log = True
):
    """
    A convenience wrapper for converting a text corpus to
    an SRP collection as a block.

    inputfile is the collection. The format is the same as the ingest
    format used by bookworm and mallet; that is to say, a single unicode
    file where each line is a document represented as a unique filename, then a tab,
    and then a long string containing the text of the document.
    To coerce into the this format, newlines must be removed.

    inputfile can also be a **directory** of txt files.

    outputfile is the SRP file to be created. Recommended suffix is `.bin`.

    dims is the dimensionality of the output SRP.

    """
    import SRP

    output = Vector_file(outputfile, dims=dim, mode="w")
    hasher = SRP.SRP(dim=dim, )

    if inputfile.endswith(".txt"):
        yielder = textset_yielder
    elif os.path.isdir(inputfile):
        yielder = directory_yielder
    else:
        raise ValueError("Don't know how to process {}: must be a textfile or a directory".format(inputfile))
    
    for i, (id, txt) in enumerate(yielder(inputfile)):
        transform = hasher.stable_transform(txt, log=True, standardize=True)
        output.add_row(id, transform)
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

    def __init__(self, filename, dims=float("Inf"), mode="r", max_rows=float("Inf"), precision = 4):
        """
        Creates an SRP object.

        filename: The location on disk.
        dims: The number of vectors to store for each document. Typically ~100 to ~1000.
              Need not be specified if working with an existing file.
        mode: One of: 'r' (read an existing file); 'w' (create a new file); 'a' (append to the 
              end of an existing file)
        max_rows: clip the document to a fixed length. Best left unused.
        precision: bytes to use for each 
        """
        
        self.filename = filename
        self.dims = dims
        self.mode = mode
        self.max_rows = max_rows
        if not precision in [2, 4]:
            e = "Only `4` (single) and `2` (half) bytes are valid options for `precision`"
            raise ValueError(e)
        self.precision = precision
        self.float_format = '<f{}'.format(precision)
        if self.mode == "r":
            self._open_for_reading()
        if self.mode == "w":
            self._open_for_writing()
        if self.mode == "a":
            self._open_for_appending()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
            
    def repair_file(self):
        """
        When writing millions of these, sometimes bytes get unaligned: this
        is an imprecise rubric that generally works to fix corrupted data.
        """
        if self.mode != "a":
            raise IOError("Can only repair when in append mode")
        
        # This also sets the pointer to the front.
        self._preload_metadata()
        self.nrows = 0
        # Avoid breaking the loop.

        previous_start = self.file.tell()

        while True:
            try:
                _, _ = self._next_line()
            except StopIteration:
                break
            except RuntimeError:
                break
            except UnicodeDecodeError:
                break
            except ValueError:
                break
            self.nrows += 1
            if self.nrows % 100000 == 0:
                print("{} read".format(self.nrows))
            previous_start = self.file.tell()
        print("Recovered {} rows".format(self.nrows))
        self.file.truncate(previous_start)
        self._rewrite_header()
        

    def concatenate_file(self, filename):
        """
        Adds all record in a different file to the end of this one.
        Useful if creation of files has been parallelized or distributed.

        In theory, it should be possible to add a large file to a smaller one.
        """

        with Vector_file(filename, dims=self.dims, mode="r", precision = self.precision) as new_file:
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
        self.file.seek(0, 2)

    def _open_for_writing(self):
        self.nrows = 0
        self.file = open(self.filename, "wb")
        self._rewrite_header()
        self.vector_size = self.dims

    def _open_for_reading(self):
        self.file = open(self.filename, 'rb')
        self._preload_metadata()

    def to_matrix(self, unit_length=False, clean=False):
        """
        Returns the entire file as a matrix with names (wrapped in a dict).

        This can, obviously, overflow memory on a large file.
        """
        labels = []
        matrix = np.zeros(
            (min([self.vocab_size, self.max_rows]), self.dims), "<f4")
        for i, (id, row) in enumerate(self):
            labels.append(id)
            if unit_length:
                row = row/np.linalg.norm(row)
            matrix[i] = row
        return {"names": labels, "matrix": matrix}

    def _open_for_appending(self):
        try:
            self.file = open(self.filename, "rb+")
        except FileNotFoundError:
            return self._open_for_writing()
        self._preload_metadata()
        self.file.seek(0, 2)
        self.nrows = self.vocab_size
        if self.dims != self.vector_size:
            raise IndexError(
                "The existing files is {} dimensions: unable to append"
                " with {} dimensions as requested".format(
                    self.vector_size, self.dims))

    def _recover_from_corruption(self):
        starting = self.pos
        self.debug_mode = True
        iterator = self._iter__()
        for i in range(2000000):
            self.file.seek(starting + i)
            try:
                gah = iterator.__next__()
                safe_pos = self.pos
                for n in range(10):
                    # Make sure things are relatively straightforward from here on out.
                    _ = self.next()
                self.file.seek(safe_pos)
                del self.debug_mode
                return True
            except StopIteration:
                pass
            except RuntimeError:
                break            
        print("Encountered corrupted data with {} words left and unable to recover at all".format(
            self.remaining_words))
        raise StopIteration

    def add_row(self, identifier, array):
        """
        Add a new document/word/whatever to the matrix.
        """
        try:
            if " " in identifier:
                raise TypeError("Spaces are not allowed in row identifiers")
        except UnicodeDecodeError:
            if " " in identifier.decode("utf-8"):
                raise TypeError("Spaces are not allowed in row identifiers")
        
        if type(array) != np.ndarray:
            raise TypeError("Must pass a numpy ndarray as array")
        
        if array.dtype != np.dtype(self.float_format):
            if (array.dtype == np.dtype("<f4")) and self.precision == 2:
                array = array.astype(self.float_format)
            else:
                raise TypeError("Numpy array must be of type '<f4'")
        if len(array) != self.dims:
            raise IndexError("The existing files is {} dimensions: unable to append with {} dimensions as requested".format(
                self.vector_size, self.dims))
        
        if py2:
            try:
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
        self.vocab_size, self.vector_size = map(
            int, header.split())  # throws for invalid file format
        # Some shuffling to decide whether all the columns are going to be read.
        if self.dims < float("Inf") and self.dims > self.vector_size:
            import sys
            sys.stderr.write(
                "WARNING: data has only {} columns but call requested top {}".format(
                    self.vector_size, self.dims))
        if self.dims == float("Inf") or self.dims == self.vector_size:
            self.dims = self.vector_size
            self.slice_and_dice = False
        else:
            self.slice_and_dice = True

        self.remaining_words = min([self.vocab_size, self.max_rows])

        self._check_if_half_precision()
        
    def _check_if_half_precision(self):
        
        body_start = self.file.tell()        
        word, weights = (self._read_row_name(), self._read_binary_row())

        meanval = np.mean(np.abs(weights))
        
        if meanval > 1e10:
            warning = "Average size is extremely large" + \
               "did you mean to specify 'precision = 2'?"
            warnings.warn(warning)
        
        self.file.seek(body_start)

        
    def _read_row_name(self):
        buffer = []
        while True:
            ch = self.file.read(1)
            if not ch:
                print("Ran out of data with {} words left".format(
                    self.remaining_words))
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
            if not hasattr(self, "debug_mode"):
                self._recover_from_corruption()
            else:
                raise
        except:
            print("Couldn't export:")
            print(word)
            raise
        return word

    def _build_offset_lookup(self, force=False):
        if hasattr(self, "_offset_lookup") and not force:
            return
        self._offset_lookup = {}
        self._preload_metadata()
        # Add warning for duplicate ids.
        i = 0
        while i < self.vocab_size:
            label = self._read_row_name()
            if label in self._offset_lookup:
                warnings.warn(
                    "Warning: this vector file has duplicate identifiers " + 
                    "(words) The last vector representation of each " + 
                    "identifier will be used, and earlier ones ignored.")
            self._offset_lookup[label] = self.file.tell()
            # Skip to the next name without reading.
            self.file.seek(self.precision*self.vector_size, 1)
            i += 1

    def sort(self, destination, sort = "names", safe = True, chunk_size = 2000):
        """
        This method sorts a vector file by its keys without reading it into memory.

        It also cleans 

        destination: A new file to be written.

        sort: one of 'names' (default sort by the filenames), 'random' 
        (sort randomly), or 'none' (keep the current order)

        safe: whether to check for (and eliminate) duplicate keys and 

        chunk_size: How many vectors to read into memory at a time. Larger numbers
        may improve performance, especially on hard drives,
        by keeping the disk head from moving around.
        """
        
        self._build_offset_lookup()
        ks = list(self._offset_lookup.keys())
        if sort == 'names':
            ks.sort()
        elif sort == 'random':
            random.shuffle(ks)
        elif sort == 'none':
            pass
        else:
            raise NotImplementedError("sort type must be one of [names, random, none]")
        # Chunk size matters because we can pull the vectors
        # from the disk in order within each chunk.

        last_written = None
        with Vector_file(destination,
                         dims = self.dims,
                         mode = "w",
                         precision = self.precision) as output:
            for i in range(0, len(ks), chunk_size):
                keys = ks[i:(i + chunk_size)]
                for key, row in zip(keys, self[keys]):
                    if safe:
                        norm = np.linalg.norm(row)
                        if np.isinf(norm) or np.isnan(norm) or norm == 0:
                            continue
                        if key == last_written:
                            continue
                    last_written = key
                    output.add_row(key, row)

    def _regex_search(self, regex):
        
        self._build_offset_lookup()
        values = [(i, k) for k, i in self._offset_lookup.items() if re.search(regex, k)]
        # Sort to ensure values are returned in disk order.
        values.sort()
        for i, k in values:
            yield (k, self[k])
        
    def __getitem__(self, label):
        """
        Attributes can be accessed in three ways.
    

        With a string: this returns just the vector for that string.
        With a list of strings: this returns a multidimensional array for each query passed.
          If any of the requested items do not exist, this will fail.
        With a single *compiled* regular expression (from either the regex or re module). This
          will return an iterator over key, value pairs of keys that match the regex.
        """
        self._build_offset_lookup()
        
        if isinstance(label, original_regex_type):
            label = re.compile(label.pattern)
            
        if isinstance(label, regex_type):
            return self._regex_search(label)
        
        if isinstance(label, MutableSequence):
            is_iterable = True
        else:
            is_iterable = False
            label = [label]

        vecs = []
        # Will fail on any missing labels

        # Prefill and sort so that any block are done in disk-order.
        # This may make a big difference if you're on a tape drive!
        
        vecs = np.zeros((len(label), self.vector_size), '<f4')
        elements = [(self._offset_lookup[l], i) for i, l in enumerate(label)]
        elements.sort()

        for offset, i in elements:
            self.file.seek(offset)            
            vecs[i] = self._read_binary_row()

    
        if is_iterable:
            return np.stack(vecs)
        else:
            return vecs[0]

    def _read_binary_row(self):
        binary_len = self.precision * self.vector_size
        self.pos = self.file.tell()
        if self.slice_and_dice:
            # When dims is less than the resolution of the file size.
            read_length = self.precision*self.dims
            weights = np.frombuffer(self.file.read(read_length), dtype=self.float_format)
            # Catch up the pointer.
            _ = self.file.read(self.precision*self.vector_size - read_length)
        else:
            try:
                weights = np.frombuffer(
                    self.file.read(binary_len), dtype=self.float_format)
            except ValueError:
                print("Can't parse data with {} words left".format(
                    self.remaining_words))
                raise StopIteration
            if len(weights) != self.vector_size:
                print("Ran out of data with {} words left".format(
                    self.remaining_words))
                raise StopIteration
        if self.mode=='r' and self.precision == 2:
            weights = weights.astype("<f4")
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
            if self.remaining_words <= -1:
                # Allow breaking out of the loop early.
                return
            yield self._next_line()

    def _next_line(self):
        word = self._read_row_name()
        weights = self._read_binary_row()        
        return (word, weights)

if __name__ == "__main__":
    run_arguments()
