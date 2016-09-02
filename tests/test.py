# -*- coding: utf-8 -*-

import SRP
import numpy as np
import unittest

class Bookworm_SQL_Creation(unittest.TestCase):
    def test_ascii(self):
        hasher = SRP.SRP(6)
        self.assertEqual(
            hasher.stable_transform("hello world", log=False).tolist(),
            np.array([0.,  0.,  2.,  0.,  2.,  0.]).tolist()
            )

    def wordcounts_unicode(self):
        hasher = SRP.SRP(160)
        
        wordcount_style = hasher.stable_tranform(
            words = [u"Güten",u"Tag"],
            counts = [1,1],
            log=False
        )

        string_style = hash.stable_transform(
            words =  u"Güten Tag",
            log=False
        )

        self.assertEqual(wordcount_style,string_style)
        
    def test_unicode(self):
        hasher = SRP.SRP(6)
        guten = u"Güten Tag"
        gutenhash = np.array([0., 2., -2., 0., 2.,0.]).tolist()

        basic = hasher.stable_transform(guten).tolist()
        self.assertTrue(basic == gutenhash)

        encoded = hasher.stable_transform(guten.encode("utf-8"),log=False).tolist()
        self.assertTrue(encoded == gutenhash)

        decoded = hasher.stable_transform(guten.encode("utf-8").decode("utf-8"),log=False).tolist()
        self.assertTrue(decoded == gutenhash)

        
if __name__=="__main__":
    unittest.main()
