from glob import glob
import subprocess
import re
import dateparser
import csv


DATE_CLUES = ['statement date', 'as at']


def parse_statement_date(line):
    line_lower = line.lower()
    for clue in DATE_CLUES:
        if clue in line_lower:
            statement_date = dateparser.parse(line_lower.split(clue)[-1])
            if statement_date:
                return statement_date


def process_pdf(filename, csv_writer):
    print(filename)
    result = subprocess.run(['pdftotext', '-layout', filename, '-'],
                            stdout=subprocess.PIPE)
    lines = result.stdout.decode('utf-8').split('\n')
    statement_date = None
    for line in lines:
        if not line:
            continue
        # this assumes that statement date is always before transaction records
        if not statement_date:
            statement_date = parse_statement_date(line)
        groups = re.split(r'\s{2,}', line.strip())
        if not groups or len(groups) < 3 or len(groups[0]) < 5:
            continue
        date_found = dateparser.parse(groups[0])
        if date_found:
            description_end_index = -1
            if '$' in groups:
                # everything between date & $ is considered description
                description_end_index = groups.index('$')
            groups = [groups[0], ' '.join(groups[1:description_end_index]), groups[-1],
                      statement_date, filename]
            csv_writer.writerow(groups)


if __name__ == '__main__':
    with open('out_wei_test.csv', 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        for filename in glob('/Users/luwei/drive/CS4225_project/data/pdf/*.pdf'):
            process_pdf(filename, csv_writer)
            break
