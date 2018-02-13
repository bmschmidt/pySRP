#### -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from future.utils import iteritems

import SRP
import numpy as np
import unittest

class ReadAndWrite(unittest.TestCase):
    array1 = np.array([1,2,3],'<f4')
    array2 = np.array([3,2,1],'<f4')
    test_set = [("foo",array1),
                ("foo2",array1),
                ("fü",array2),
                ("stop",array2)]
        

    def test_creation_and_reading(self):
        testfile = SRP.Vector_file("test.bin", dims=3, mode="w")
        for row in self.test_set:
            if row[0] == "stop":
                continue
            testfile.add_row(*row)

        self.assertTrue(testfile.nrows==3)
        testfile.close()

        testfile2 = SRP.Vector_file("test.bin", dims=3, mode="a")
        testfile2.add_row(*self.test_set[3])
        testfile2.close()
        self.assertTrue(testfile2.nrows==4)

        
        foo = SRP.Vector_file("test.bin",mode="r")
        self.assertTrue(foo.vocab_size==4)

        read_in_values = dict()
        for (i,(name,array)) in enumerate(foo):
            read_in_values[name] = array
            (comp_name,comp_array) = self.test_set[i]
            self.assertEqual(comp_name,name)
            self.assertEqual(array.tolist(),comp_array.tolist())
            
        foo.close()
        
        self.assertEqual(read_in_values["foo"].tolist(),read_in_values["foo2"].tolist())
        self.assertFalse(read_in_values["foo2"].tolist()==read_in_values["fü"].tolist())

        
class BasicHashing(unittest.TestCase):
    def test_ascii(self):
        hasher = SRP.SRP(6)
        hello_world = hasher.stable_transform("hello world", log=False)

        self.assertEqual(
            hello_world.tolist(),
            np.array([0.,  0.,  2.,  0.,  2.,  0.]).tolist()
            )

    def test_wordcounts_unicode(self):
        hasher = SRP.SRP(160)

        wordcount_style = hasher.stable_transform(
            words = [u"Güten",u"Tag"],
            counts = [1,1],
            log=False
        ).tolist()

        string_style = hasher.stable_transform(
            words =  u"Güten Tag",
            log=False
        ).tolist()
    
        self.assertEqual(wordcount_style,string_style)
        
    def test_ascii_equals_unicode(self):
        hasher = SRP.SRP(160)
        
        hello_world = hasher.stable_transform("hello world", log=False).tolist()
        hello_world_unicode = hasher.stable_transform(u"hello world", log=False).tolist()
        
        self.assertEqual(hello_world,hello_world_unicode)

    def test_logs_are_plausible(self):
        log_unit = np.log(1e05)

        hasher = SRP.SRP(20)
        log_srp = hasher.stable_transform("hello", log=True)
        nonlog_srp = hasher.stable_transform("hello", log=False)
        difference = sum(log_srp - (nonlog_srp) * log_unit)
        
        # Forgive floating point error.
        self.assertTrue(difference < 1e-05)
        
    def test_unicode(self):
        """
        One of the goals is be able to pass *either* encoded or decoded
        utf-8, because that tends to happen.

        """
        hasher = SRP.SRP(6)
        guten = u"Güten Tag"
        gutenhash = np.array([0., 2., -2., 0., 2.,0.]).tolist()

        basic = hasher.stable_transform(guten, log=False).tolist()
        self.assertTrue(basic == gutenhash)

        encoded = hasher.stable_transform(guten.encode("utf-8"), log=False).tolist()
        self.assertTrue(encoded == gutenhash)

        decoded = hasher.stable_transform(guten.encode("utf-8").decode("utf-8"),log=False).tolist()
        self.assertTrue(decoded == gutenhash)

    def test_standardization(self):
        """
        standardization does case normalization,
        and tokenizes by a charater regex.
        """
        hasher = SRP.SRP(6)
        string1 = "Gravity's rainbow"
        hashed_standardized = hasher.stable_transform(string1, log=False, standardize=True)
        manually_tokenized = ["Gravity","s","RAINBOW"]
        hashed_manually_tokenized = hasher.stable_transform(manually_tokenized, [1, 1, 1], log=False,standardize=True)
        self.assertEqual(hashed_manually_tokenized.tolist(), hashed_standardized.tolist())

if __name__=="__main__":
    unittest.main()
