import bnelearn

def test_is_string():
    s=bnelearn.joke()
    assert isinstance(s, str)