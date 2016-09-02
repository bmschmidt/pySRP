# pySRP
Python Module implementing SRP

Note: Pre-alpha. The API is still unstable.

## Usage

```python
import SRP

# initialize with desired number of dimensions
hasher = SRP.SRP(640)

```

The most important argument is 'stable_transform'.

This can hash strings

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

Not yet formally implemented.
