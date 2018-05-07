import unittest
import csv
import io
import difflib
import os
import time
from datetime import datetime
from glob import glob
from pprint import pprint
from pdftotxt import process_pdf, parse_transaction_date

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


    def test_parse_transaction_date(self):
        statement_date = datetime(2016, 4, 1)
        date = parse_transaction_date('24/03', statement_date)
        self.assertEqual(date, '2016-03-24')

        statement_date = datetime(2018, 1, 1)
        date = parse_transaction_date('24/12', statement_date)
        self.assertEqual(date, '2017-12-24')

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
