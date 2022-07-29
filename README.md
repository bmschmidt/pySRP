# pySRP

Python module implementing Stable Random Projections, as described in 
[Cultural Analytics Vol. 1, Issue 2, 2018 October 04, 2018: Stable Random Projection: Lightweight, General-Purpose Dimensionality Reduction for Digitized Libraries](https://doi.org/10.22148/16.025)

These create interchangeable, data-agnostic vectorized representations of text suitable for a variety of contexts. Unlike most vectorizations, they are suitable for representing texts in any language that uses space-tokenization, or non-linguistic content, since they contain no implicit language model besides words.

You may want to use them in concert with the pre-distributed Hathi SRP features
described further here.

## Installation

Requires python 3

```bash
pip install pysrp
```
## Changelog

**Version 2.0 (July 2022) slightly changes the default tokenization algorithm!**

Previously it was `\w`; now it is `[\p{L}\p{Pc}\p{N}\p{M}]+`. 

I also no longer recommend use 

## Usage

### Examples

See the [docs folder](https://github.com/bmschmidt/pySRP/tree/master/docs)
for some IPython notebooks demonstrating:

* [Taking a subset of the full Hathi collection (100,000 works of fiction) based on
identifiers, and exploring the major clusters within fiction.](https://github.com/bmschmidt/pySRP/blob/master/docs/Build%20Fiction%20Set.ipynb)
* [Creating a new SRP representation of text files and plotting dimensionality reductions of them by language and time](https://github.com/bmschmidt/pySRP/blob/master/docs/Hash%20a%20corpus%20of%20text%20files%20into%20SRP%20space.ipynb)
* [Searching for copies of one set of books in the full HathiTrust collection, and using Hathi metadata to identify duplicates and find errors in local item descriptions.](https://github.com/bmschmidt/pySRP/blob/master/docs/Find%20Text%20Lab%20Books%20in%20Hathi.ipynb)
* [Training a classifier based on library metadata using TensorFlow, and then applyinig that classification to other sorts of text.](https://github.com/bmschmidt/pySRP/blob/master/docs/Classification%20Using%20Tensorflow%20Estimators.ipynb)

### Basic Usage

Use the SRP class to build an object to perform transformations.

This is a class method, rather than a function, which builds a cache of previously seen words.

```python
import srp
# initialize with desired number of dimensions
hasher = srp.SRP(640)
```

The most important method is 'stable_transform'.

This can tokenize and then compute the SRP.

```python
hasher.stable_transform(words = "foo bar bar"))
```

If counts are already computed, word and count vectors can be passed separately.

```python
hasher.stable_transform(words = ["foo","bar"],counts = [1,2])
```


## Read/write tools

SRP files are stored in a binary file format to save space. 
This format is the same used by the binary word2vec format.

**DEPRECATION NOTICE**

This format is now deprecated--I recommend the Apache Arrow binary serialization format instead. 

```python
file = SRP.Vector_file("hathivectors.bin")

for (key, vector) in file:
  pass
  # 'key' is a unique identifier for a document in a corpus
  # 'vector' is a `numpy.array` of type `<f4`.
```

There are two other methods. One lets you read an entire matrix in at once.
This may require lots of memory. It returns a dictionary with two keys: 'matrix' (a numpy array)
and 'names' (the row names).

```python
all = SRP.Vector_file("hathivectors.bin").to_matrix()
all['matrix'][:5]
all['names'][:5]
```

The other lets you treat the file as a dictionary of keys. The first lookup
may take a very long time; subsequent lookups will be fast *without* requiring
you to load the vectors into memory. To get a 1-dimensional representation of a book:

```python
all = SRP.Vector_file("hathivectors.bin")
all['gri.ark:/13960/t3032jj3n']
```

You can also, thanks to Peter Organisciak, access multiple vectors at once this way by passing a list of identifiers. This returns a matrix with shape (2, 160) for a 160-dimensional representation.

```python
all[['gri.ark:/13960/t3032jj3n', 'hvd.1234532123']]
```

### Writing to SRP files

You can build your own files row by row.

```python

# Note--the dimensions of the file and the hasher should be equal.
output = SRP.Vector_file("new_vectors.bin",dims=640,mode="w")
hasher = SRP.SRP(640)


for filename in [a,b,c,d]:
  hash = hasher.stable_transform(" ".join(open(filename).readlines()))
  output.add_row(filename,hash)

# files must be closed.
output.close()
```

Since files must be closed, it can be easier to use a context manager:

```python

# Note--the dimensions of the file and the hasher should be equal.
hasher = SRP.SRP(640)

with SRP.Vector_file("new_vectors.bin",dims=640,mode="w") as output:
  for filename in [a,b,c,d]:
    hash = hasher.stable_transform(" ".join(open(filename).readlines()))
    output.add_row(filename,hash)
```


