# pySRP
Python Module implementing Stable Random Projection.

Note: Pre-alpha. The API is still unstable.

## Usage

```python
import SRP

# initialize with desired number of dimensions
hasher = SRP.SRP(640)

```

The most important method is 'stable_transform'.

This can tokenize and then compute the SRP.

```python
hasher.stable_transform(words = "foo bar bar",log=True,standardize=True)
```

If counts are already computed, word and count vectors can be passed separately.

```python
hasher.stable_transform(words = ["foo","bar"],counts = [1,2],log=True,standardize=True)
```

It's worth using 'standardize'

## Read/write tools

SRP files are stored in a binary file format to save space. 
This format is the same used by the binary word2vec format.

```python
file = SRP.SRP_file("hathivectors.bin")

for (key,vector) in file:
  pass
  # 'key' is a unique identifier for a document in a corpus
  # 'vector' is a `numpy.array` of type `<f4`.

```

### Writing to SRP files

```python

# Note--the dimensions of the file and the hasher should be equal.

output = SRP.SRP_file("new_vectors.bin",dims=640,mode="w")
hasher = SRP.SRP(640)


for filename in [a,b,c,d]:
  hash = hasher.stable_transform(" ".join(open(filename).readlines()))
  output.add_row(filename,hash)
  


```
