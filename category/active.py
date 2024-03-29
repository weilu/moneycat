import csv
import json
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from os import path


def get_subcategory_to_category_map():
    all_subcats = {} # sub -> cat
    with open(path.join(path.dirname(__file__), 'categories.csv')) as f:
        reader = csv.reader(f)
        next(reader) # skip header line
        for row in reader:
            all_subcats[row[1]] = row[0]
    return all_subcats


def get_active_subcategories(subcats):
    all_subcats = get_subcategory_to_category_map()
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

    return active_subcats


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Input csv file or directory. When this argument is specified, the command outputs only the active subcategories that appear in the specified file(s).")
    parser.add_argument("-o", "--output", dest="output",
                        help="output json filename", default="categories.json")
    args = parser.parse_args()

    if not args.input:
        subcats = get_subcategory_to_category_map()
    else:
        subcats = set({})
        if path.isdir(args.input):
            for filename in glob('{}/*.csv'.format(args.input)):
                subcats.update(pd.read_csv(filename)['category'])
        else:
            subcats.update(pd.read_csv(filename)['category'])

        subcats = get_active_subcategories(subcats)

    with open(args.output, 'w') as f:
        f.write(json.dumps(subcats, indent=2, sort_keys=True))
