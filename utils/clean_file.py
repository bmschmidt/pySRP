import SRP
from SRP import Vector_file
import sys

if len(sys.argv) != 3:
    print "Usage: python clean_file.py INPUT OUTPUT"
    print "where input is a file to be stripped of crufy rows."    


fin = Vector_file(sys.argv[1])

fout = Vector_file(sys.argv[2], dims = fin.dims, mode="w")


for i, (id, row) in enumerate(fin):
    if sum(row) != 0 and abs(sum(row)) < 1e15:
        fout.add_row(id,row)
    else:
        print "Skipping row {} of magnitude {}".format(i, sum(row))
        
fout.close()
