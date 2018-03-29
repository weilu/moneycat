import pandas as pd


def parse_number(num_str):
    return float(num_str.replace('CR', '').strip())

def attempt_merge(all_txs, dbs_labeled_txs):
    # separate all_txs to DBS and UOB
    uob_condition = all_txs.source.str.startswith('eStatement_')
    uob_txs = all_txs[uob_condition]
    dbs_txs = all_txs[~uob_condition]
    # compare row counts
    print('dbs_txs: {}, dbs_labeled_txs: {}'.format(dbs_txs.shape, dbs_labeled_txs.shape))
    # sort dbs_txs and dbs_labeled_txs by date & description & amount
    dbs_txs.sort_values(['date', 'description', 'amount'], inplace=True)
    dbs_labeled_txs.sort_values(['DATE', 'TITLE', 'PRICE'], inplace=True)
    # iterate over both to merge
    dbs_txs_iter = dbs_txs.itertuples()
    n = 0
    discrepancy = 0
    for labeled_row in dbs_labeled_txs.itertuples():
        row = next(dbs_txs_iter)
        same_date = row.date == labeled_row.DATE.replace('-', ' ')
        same_desc = row.description == labeled_row.TITLE
        same_amount = (row.amount == labeled_row.PRICE \
                       or parse_number(row.amount) == parse_number(labeled_row.PRICE))
        if same_date and same_date and (not same_amount):
            discrepancy += 1
            print(row.source, row.description)
            print('  wei: {} joddiy: {}\n'.format(row.amount, labeled_row.PRICE))
        n+=1
        if n > 20:
            break
    print('{}/20 transactions have amount discrepancy'.format(discrepancy))


if __name__ == '__main__':
    all_txs = pd.read_csv('out_wei.csv', sep=",")
    dbs_labeled_txs = pd.read_csv('statementsLabeled.csv', sep=",")
    print('all_txs: {}, dbs_labeled_txs: {}'.format(all_txs.shape, dbs_labeled_txs.shape))
    # attempt_merge(all_txs, dbs_labeled_txs)

    classification = {}
    for labeled_row in dbs_labeled_txs.itertuples():
        classification[labeled_row.TITLE] = labeled_row.CATEGORY
    print('labelled dict based on description: {}'.format(len(classification)))

    categories = []
    for index, row in all_txs.iterrows():
        categories.append(classification.get(row[1]))
    all_txs['category'] = categories

    all_txs.to_csv('out_wei_labelled.csv')

