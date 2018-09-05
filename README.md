# pySRP

Python Module implementing Stable Random Projections.

These create interchangeable, data-agnostic vectorized representations of text suitable for a variety of contexts.

You may want to use them in concert with the pre-distributed Hathi SRP features.

## Usage

Use the SRP class to build an object to perform transformations.

This is a class method, rather than a function, which builds a cache of previously seen words.

```python
import SRP
# initialize with desired number of dimensions
hasher = SRP.SRP(640)
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

```python
file = SRP.Vector_file("hathivectors.bin")

for (key,vector) in file:
  pass
  # 'key' is a unique identifier for a document in a corpus
  # 'vector' is a `numpy.array` of type `<f4`.
```

There are two other methods. One lets you read an entire matrix in at once.
This may require lots of memory.

```python
all = SRP.Vector_file("hathivectors.bin").to_matrix()
all['matrix'][:5]
all['names'][:5]
```

The other lets you treat the file as a dictionary of keys. The first lookup
may take a very long time; subsequent lookups will be fast *without* requiring
you to load the vectors into memory.

```python
all = SRP.Vector_file("hathivectors.bin")
all['gri.ark:/13960/t3032jj3n']
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
