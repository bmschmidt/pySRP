import SRP
from SRP import Vector_file
import sys
import numpy as np


if not len(sys.argv) >= 3:
    print "Usage: python expand_half-precision .py INPUT OUTPUT [lengthout]"
    print "where input is a two-byte file, and OUTPUT is the desired 4-byte file."
    print "lengthout optionally lets you only write the first n lines of the file."    


fin = Vector_file(sys.argv[1], precision = 2)

fout = Vector_file(sys.argv[2], dims = fin.dims, mode="w", precision = 4)

if len(sys.argv)==4:
    limit = int(sys.argv[3])
else:
    limit = float("Inf")
    
for i, (id, row) in enumerate(fin):
    if sum(row) != 0 and np.sum(np.abs(row)) < 1e10:
        fout.add_row(id,row)
    else:
        print "Skipping row {} of magnitude {}".format(i, np.sum(np.abs(row)))
    if i >= limit - 1:
       break
       
fout.close()
