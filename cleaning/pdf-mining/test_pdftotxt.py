import unittest
import csv
import io
import difflib
import os
import time
from glob import glob
from pprint import pprint
from pdftotxt import process_pdf

SUPPORTED_BANKS = ['dbs', 'uob', 'ocbc', 'anz']

class TestPdftotxt(unittest.TestCase):

    def test_process_pdf(self):
        for bank in SUPPORTED_BANKS:
            maybe_create_expected_csv(bank)
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


# This is for development purposes only
# To add support for a new bank, this is invoked.
# Make sure to manually check the output in the csv & modify pdftotxt.py if necessary
def maybe_create_expected_csv(bank):
    expected_filename = f'data/{bank}.csv'
    if not os.path.exists(expected_filename):
        with open(expected_filename, 'w') as f:
            csv_writer = csv.writer(f)
            process_pdf(f'./data/{bank}.pdf', csv_writer)

if __name__ == '__main__':
    unittest.main()
