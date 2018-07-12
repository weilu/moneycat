import csv
import json
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from os import path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input", help="Input csv file or directory")
    parser.add_argument("-o", "--output", dest="output",
                        help="output json filename", default="categories.json")
    args = parser.parse_args()

    subcats = set({})
    if path.isdir(args.input):
        for filename in glob('{}/*.csv'.format(args.input)):
            subcats.update(pd.read_csv(filename)['category'])
    else:
        subcats.update(pd.read_csv(filename)['category'])

    all_subcats = {} # sub -> cat
    with open('categories.csv') as f:
        reader = csv.reader(f)
        next(reader) # skip header line
        for row in reader:
            all_subcats[row[1]] = row[0]
    all_cats = set(all_subcats.values())

    # look up active subcategory in sub -> cat mapping
    active_subcats = {} # sub -> cat
    for subcat in subcats:
        if subcat in all_subcats:
            active_subcats[subcat] = all_subcats[subcat]
        elif subcat in all_cats:
            pass
        else:
            print('Missing subcategory: {}'.format(subcat))

    with open(args.output, 'w') as f:
        f.write(json.dumps(active_subcats, indent=2, sort_keys=True))
