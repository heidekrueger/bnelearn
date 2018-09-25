from unittest import TestCase

import bnelearn

class TestDummy(TestCase):
    def test_is_string(self):
        s=bnelearn.joke()
        self.assertTrue(isinstance(s, str))