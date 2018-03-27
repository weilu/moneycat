from glob import glob
import subprocess
import re
import dateparser
import csv



if __name__ == '__main__':
    with open('assets/out_wei.csv', 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        for filename in glob('/Users/luwei/drive/CS4225_project/data/pdf/*.pdf'):
            print(filename)
            result = subprocess.run(['pdftotext', '-layout', filename, '-'],
                                    stdout=subprocess.PIPE)
            lines = result.stdout.decode('utf-8').split('\n')
            for line in lines:
                if not line:
                    continue
                groups = re.split(r'\s{2,}', line.strip())
                if not groups or len(groups) < 3 or len(groups[0]) < 5:
                    continue
                date_found = dateparser.parse(groups[0])
                if date_found:
                    description_end_index = -1
                    if '$' in groups:
                        # everything between date & $ is considered description
                        description_end_index = groups.index('$')
                    groups = [groups[0], ' '.join(groups[1:description_end_index]), groups[-1], filename]
                    csv_writer.writerow(groups)
