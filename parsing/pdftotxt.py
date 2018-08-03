from argparse import ArgumentParser
from glob import glob
import subprocess
import re
import dateparser
import csv
from iso4217 import Currency
from itertools import tee
from os import path
import logging


DATE_CLUES = ['statement date', 'as at']
CURRENCIES = set([c.code for c in Currency])
CURRENCY_AMOUNT_REGEX = '({} \d+[\.|,|\d]*\d+)'
OCBC_STATEMENT_DATE_REGEX = '\d{2} [a-z]{3,} \d{4}'
SC_STATEMENT_STOPWORD = 'To be continued...'
LANGUAGES = ['en']
INCORRECT_PWD = 'Incorrect password'


def parse_statement_date(line, iterator):
    line_lower = line.lower()
    # special case for ocbc bank statement
    if re.match(OCBC_STATEMENT_DATE_REGEX, line_lower):
        return parse_date(line_lower)
    for clue in DATE_CLUES:
        if clue in line_lower:
            text_after_date = line_lower.split(clue)[-1]
            for group in split_line(text_after_date):
                statement_date = parse_date(re.sub(r'[^\w\s]', '', group))
                if statement_date:
                    return statement_date
            # in OCBC & ANZ's case, need to look at the next non-empty line
            if not statement_date:
                line = next(iterator).strip()
                while not line:
                    line = next(iterator).strip()
                groups = split_line(line)
                if not groups:
                    return
                statement_date = parse_date(groups[0])
                if statement_date:
                    return statement_date


# "(1,380.77)" and "1,380.77 CR" will translate to -1380.77
def parse_amount(amount_str):
    try:
        amount_str = amount_str.lower().strip().replace(',', '')
        cleaned = re.sub(r'\(|\)|cr', '', amount_str).strip()
        amount = float(cleaned)
        if cleaned != amount_str:
            amount = -amount
        return amount
    except ValueError:
        return None


# cross year statement needs statement_date to determine the year of date_str
# because transaction date_str often don't contain year
def parse_transaction_date(date_str, statement_date):
    date_found = parse_date(date_str)
    if not date_found:
        return None
    transaction_date = date_found.replace(year=statement_date.year)
    alt_date = transaction_date.replace(year=(statement_date.year-1))
    if (abs(alt_date - statement_date) < abs(transaction_date - statement_date)):
        transaction_date = alt_date
    return format_date(transaction_date)


def format_date(datetime_obj):
    return datetime_obj.strftime('%Y-%m-%d')


def parse_date(date_str):
    return dateparser.parse(date_str, locales=['en-SG'])


# Foreign currency transaction often include the foreign currency &
# amount in a separate line
def peek_forward_for_currency_and_description(iterator,
        max_lines=2, next_tx_condition=None):
    if not next_tx_condition:
        next_tx_condition = lambda groups: (groups and parse_date(groups[0]))
    foreign_amount = ''
    descriptions = []
    for i in range(0, max_lines): # look no further than 2 lines
        line = next(iterator).strip()
        if line and len(line) > 3:
            if next_tx_condition(split_line(line)):
                break # break early if found the next transaction line
            descriptions.append(line)
            if not foreign_amount:
                for currency in CURRENCIES:
                    found = re.search(CURRENCY_AMOUNT_REGEX.format(currency),
                                      line, re.IGNORECASE)
                    if found:
                        foreign_amount = found.group(1)
                        break
        else:
            break # also break early on blank line
    return foreign_amount, ' '.join(descriptions)


def signed_tx_amount(amount, amount_index, deposit_start_index,
                     withdrawal_start_index):
    min_index = min(deposit_start_index, withdrawal_start_index)
    max_index = max(deposit_start_index, withdrawal_start_index)
    is_first_column = (amount_index >= min_index and amount_index < max_index)
    deposit_first = (min_index == deposit_start_index)
    # is_first_column | deposit_first | +/- (where + is spend, - is deposit)
    # T               | T             | -
    # T               | F             | +
    # F               | T             | +
    # F               | F             | -
    if is_first_column ^ deposit_first:
        return amount
    else:
        return -amount


def split_line(line):
    return re.split(r'\s{2,}', line)

def process_pdf(filename, csv_writer, pdftotxt_bin='pdftotext',
                include_source=True, password=None, **kwargs):

    # recursive
    def process_bank_statement_line(iterator, statement_date, header_line,
                                    prev_tx_date=None):
        deposit_start_index = header_line.index('deposit')
        withdrawal_start_index = header_line.index('withdrawal')

        # TODO: make it DRY
        try:
            # skip empty & short lines
            line, line_stripped, groups = None, None, None
            while not line_stripped or not groups or len(groups) < 3 or len(groups[0]) < 5:
                line = next(iterator)
                line_stripped = line.strip()
                if line_stripped:
                    line_lower = line.lower()
                    if 'deposit' in line_lower and 'withdrawal' in line_lower:
                        return process_bank_statement_line(iterator,
                                statement_date, line_lower)
                    groups = split_line(line_stripped)
                    if not statement_date:
                        statement_date = parse_statement_date(line_stripped,
                                                              iterator)

            # consider a line as a transaction when it has 3 groups or more & ends with 2 numbers
            tx_amount = parse_amount(groups[-2])
            balance_amount = parse_amount(groups[-1])
            if tx_amount != None and balance_amount != None:
                tx_date = parse_transaction_date(groups[0], statement_date)
                if not tx_date:
                    tx_date = prev_tx_date
                    description_start = 0
                else:
                    prev_tx_date = tx_date
                    # everything between last occurance of tx_date and
                    # 2nd last number is considered description
                    description_start = line.rfind(groups[0]) + len(groups[0])
                description_end = line.index(groups[-2])
                description = line[description_start:description_end].strip()
                amount = signed_tx_amount(tx_amount, line.index(groups[-2]),
                        deposit_start_index, withdrawal_start_index)

                def should_stop_peeking(groups):
                    if len(groups) >= 3:
                        if parse_date(groups[0]):
                            return True
                        if parse_amount(groups[-2]) != None and \
                           parse_amount(groups[-1]) != None:
                            return True
                    elif len(groups) >=1 and SC_STATEMENT_STOPWORD in groups[0]:
                        return True
                    else:
                        return False
                # make a copy for peeking
                iterator, iterator_copy = tee(iterator)
                foreign_amount, more_desc = peek_forward_for_currency_and_description(
                        iterator_copy, max_lines=5, next_tx_condition=should_stop_peeking)
                description = re.sub(' +', ' ', f'{description} {more_desc}').strip()

                #TODO: handle balance
                row = [tx_date, description, amount, foreign_amount,
                       format_date(statement_date)]
                if include_source:
                    row.append(path.basename(filename))
                csv_writer.writerow(row)

            process_bank_statement_line(iterator, statement_date, header_line,
                                        prev_tx_date=prev_tx_date)
        except StopIteration:
            pass


    # recursive fn
    def process_line(iterator, statement_date):
        try:
            # skip empty & short lines
            line, line_stripped, groups = None, None, None
            while not line_stripped or not groups or len(groups) < 3 or len(groups[0]) < 5:
                line = next(iterator)
                line_stripped = line.strip()
                if line_stripped:
                    line_lower = line.lower()
                    if 'deposit' in line_lower and 'withdrawal' in line_lower:
                        return process_bank_statement_line(iterator,
                                statement_date, line_lower)
                    groups = split_line(line_stripped)
                    if not statement_date:
                        statement_date = parse_statement_date(line_stripped, iterator)

            # consider a line as a transaction when it begins with date
            tx_date = parse_transaction_date(groups[0], statement_date)
            if tx_date:
                description_end_index = -1
                if '$' in groups:
                    # everything between tx_date & $ is considered description
                    description_end_index = groups.index('$')
                description = ' '.join(groups[1:description_end_index])

                # make a copy for peeking
                iterator, iterator_copy = tee(iterator)
                foreign_amount, _ = peek_forward_for_currency_and_description(iterator_copy)

                amount = parse_amount(groups[-1])
                row = [tx_date, description, amount, foreign_amount,
                       format_date(statement_date)]
                if include_source:
                    row.append(path.basename(filename))
                csv_writer.writerow(row)

            process_line(iterator, statement_date)
        except StopIteration:
            pass

    print(filename)
    if password:
        command = [pdftotxt_bin, '-layout', '-upw', password, filename, '-']
    else:
        command = [pdftotxt_bin, '-layout', filename, '-']
    try:
        result = subprocess.check_output(command, stderr=subprocess.PIPE, **kwargs)
        lines = result.decode('utf-8').split('\n')
        statement_date = None
        process_line(iter(lines), statement_date)
    except subprocess.CalledProcessError as grepexc:
        err = grepexc.stderr.decode('utf-8')
        if INCORRECT_PWD in err:
            raise RuntimeError(INCORRECT_PWD)
        print("error code", grepexc.returncode, err)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input", help="Input pdf file or directory")
    parser.add_argument("-o", "--output", dest="output",
                        help="output csv filename", default="out.csv")
    args = parser.parse_args()

    with open(args.output, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['date', 'description', 'amount', 'foreign_amount',
                             'statement_date', 'source'])
        if path.isdir(args.input):
            for filename in glob('{}/*.pdf'.format(args.input)):
                process_pdf(filename, csv_writer)
        else:
            process_pdf(args.input, csv_writer)
