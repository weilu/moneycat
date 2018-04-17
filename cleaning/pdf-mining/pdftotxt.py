from glob import glob
import subprocess
import re
import dateparser
import csv
from iso4217 import Currency
from itertools import tee
from os import path


DATE_CLUES = ['statement date', 'as at']
CURRENCIES = set([c.code for c in Currency])
CURRENCY_AMOUNT_REGEX = '({} \d+[\.|,|\d]*\d+)'


def parse_statement_date(line):
    line_lower = line.lower()
    for clue in DATE_CLUES:
        if clue in line_lower:
            statement_date = dateparser.parse(line_lower.split(clue)[-1])
            if statement_date:
                return statement_date


# Foreign currency transaction often include the foreign currency &
# amount in a separate line
def peek_forward_for_currency(iterator, max_lines=2):
    for i in range(0, max_lines): # look no further than 2 lines
        line = next(iterator).strip()
        if line and len(line) > 3:
            for currency in CURRENCIES:
                found = re.search(CURRENCY_AMOUNT_REGEX.format(currency),
                                  line, re.IGNORECASE)
                if found:
                    return found.group(1)


def process_pdf(filename, csv_writer, pdftotxt_bin='pdftotext',
                include_source=True, **kwargs):

    # recursive fn
    def process_line(iterator, statement_date):
        try:
            line = next(iterator).strip()
            if not line:
                return process_line(iterator, statement_date)
            # this assumes that statement date is always before transaction records
            if not statement_date:
                statement_date = parse_statement_date(line)
            groups = re.split(r'\s{2,}', line)
            if not groups or len(groups) < 3 or len(groups[0]) < 5:
                return process_line(iterator, statement_date)

            # consider a line as a transaction when it begins with date
            date_found = dateparser.parse(groups[0])
            if date_found:
                description_end_index = -1
                if '$' in groups:
                    # everything between date & $ is considered description
                    description_end_index = groups.index('$')
                description = ' '.join(groups[1:description_end_index])

                # make a copy for peeking
                iterator, iterator_copy = tee(iterator)
                foreign_amount = peek_forward_for_currency(iterator_copy)

                row = [groups[0], description, groups[-1], foreign_amount,
                       statement_date]
                if include_source:
                    row.append(path.basename(filename))
                csv_writer.writerow(row)

            process_line(iterator, statement_date)
        except StopIteration:
            pass

    print(filename)
    result = subprocess.run([pdftotxt_bin, '-layout', filename, '-'],
                            stdout=subprocess.PIPE, **kwargs)
    lines = result.stdout.decode('utf-8').split('\n')
    statement_date = None
    process_line(iter(lines), statement_date)


if __name__ == '__main__':
    with open('out_wei.csv', 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['date', 'description', 'amount', 'foreign_amount',
                             'statement_date', 'source'])
        for filename in glob('/Users/luwei/drive/CS4225_project/data/pdf/*.pdf'):
            process_pdf(filename, csv_writer)
