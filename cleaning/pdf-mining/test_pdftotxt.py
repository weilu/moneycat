import unittest
import csv
import io
import difflib
import time
from glob import glob
from pprint import pprint
from pdftotxt import process_pdf

SUPPORTED_BANKS = ['dbs', 'uob']

class TestPdftotxt(unittest.TestCase):

    def test_process_pdf(self):
        for bank in SUPPORTED_BANKS:
            with open(f'data/{bank}.csv') as ef:
                expected = ef.readlines()
            with io.StringIO() as f:
                csv_writer = csv.writer(f)
                start = time.clock()
                process_pdf(f'./data/{bank}.pdf', csv_writer)
                print(f"Parsing {bank} statement took {time.clock() - start} second")
                f.seek(0)
                actual = f.readlines()
                self.assertFalse(diff(expected, actual))


def diff(a, b):
    stripped_a = list(map(str.strip, a))
    stripped_b = list(map(str.strip, b))
    results = list(difflib.unified_diff(stripped_a, stripped_b))
    if results:
        pprint(results)
    return results


if __name__ == '__main__':
    unittest.main()
