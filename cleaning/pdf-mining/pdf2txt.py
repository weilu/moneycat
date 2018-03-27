#!/usr/bin/env python

"""
Converts PDF text content (though not images containing text) to plain text, html, xml or "tags".
"""
import io
import json
import pandas as pd
import re

import os
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage


# Open a PDF document.


def pdf2txt(file_path):
    """
    extract txt from pdf
    :param file_path:
    :return:
    """
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(file_path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(fp, caching=True, check_extractable=True):
        interpreter.process_page(page)
    text = retstr.getvalue()
    return text


def dbs_table_parse(txt):
    """

    :param txt:
    :return:
    """
    result = []
    tmp_txt = ""
    find_begin = False
    for line in txt.split("\n"):
        if not find_begin:
            if re.search("CARD NO.:", line):
                find_begin = True
            continue
        if re.search("TOTAL", line):
            continue
        tmp_txt += line + "\n"
    all_records = []
    other_records = []
    for line in tmp_txt.split("\n\n"):
        line = line.strip().replace("\n", " ")
        if re.search("REF NO :.", line):
            # line = re.sub("REF NO :.*$", "", line)
            line = re.sub('\s+', " ", line)
            all_records.append(line)
            pass
        elif line is not "$":
            other_records.append(line)
    count_line = len(all_records)
    for i in range(count_line):
        result.append([other_records[i], all_records[i], other_records[count_line + i]])
    # print(result)
    # exit()
    return result


def uob_table_parse(txt, idx):
    """

    :param txt:
    :return:
    """
    result = []
    data_record = []
    trans_record = []
    count_record = []
    lines = []
    is_skip = True
    for line in txt.split("\n\n"):
        line = line.strip().replace("\n", " ")
        if is_skip:
            if re.search("Trans Date", line):
                is_skip = False
            continue
        if re.search("--------------------------------------------------", line):
            continue
        lines.append(line)
    for line in lines:
        # print(line)
        line = line.strip().replace("\n", " ")
        if re.search("SGD|^UOB.*$|"
                     "LU WEI|Trans Date|Post Date|Description of Transaction|Page [0-9]+ of [0-9]+",
                     line) or len(line) == 0:
            continue
        if re.search("^[0-9]{2} JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC$", line) and len(line) == 6:
            data_record.append(line)
        elif re.search("^([+-]?)((\d{1,3}(,\d{3})*)|(\d+))(\.\d{2})?(\sCR)*$", line):
            count_record.append(line)
        else:
            trans_record.append(line)
    date_idx = 0
    # print(data_record, trans_record, count_record)
    # print(len(data_record), len(trans_record), len(count_record))
    data_r = []
    trans_r = []
    for i in range(len(data_record) // 2):
        if re.search("TOTAL|PREVIOUS", trans_record[i]):
            continue
        if re.search("CHOOSE ANY 5 TRANSACTIONS ", trans_record[i]):
            break
        data_r.append(data_record[2 * date_idx])
        trans_r.append(trans_record[i])
        date_idx += 1
    for i in range(len(trans_r)):
        result.append([data_r[i], trans_r[i], count_record.pop()])
    # print(result)
    # exit()
    return result


def uob_parse(txt):
    """
    parse uob statement
    :param txt:
    :return:
    """
    result = []
    idx = 1
    txt = re.split("End of Transaction Details", txt)[0]
    for page in re.split("Please note that you", txt):
        # print(page)
        result += uob_table_parse(page, idx)
        idx += 1
    return result


def dbs_parse(txt):
    """
    parse dbs statement
    :param txt:
    :return:
    """
    result = []
    for page in re.split("PG [0-9]+ OF [0-9]+", txt):
        result += dbs_table_parse(page)
    return result


if __name__ == '__main__':
    result = []
    # dbs
    dir = "assets/dbs"
    for filename in os.listdir(dir):
        if filename.endswith(".pdf"):
            txt = pdf2txt(dir + "/" + filename)
            result += dbs_parse(txt)
            print(filename, len(result))
    # uob
    dir = "assets/uob"
    for filename in os.listdir(dir):
        if filename.endswith(".pdf"):
            txt = pdf2txt(dir + "/" + filename)
            # print(filename)
            result += uob_parse(txt)
            print(filename, len(result))
            # print(result)
    dic = {}
    list_date = []
    list_trans = []
    list_count = []
    for item in result:
        list_date.append(item[0])
        list_trans.append(item[1])
        list_count.append(item[2])
    dic["DATE"] = list_date
    dic["TITLE"] = list_trans
    dic["PRICE"] = list_count
    df = pd.DataFrame(data=dic)
    df.to_csv("out.csv", sep=',', index=False)
    # with open(dir + "/out.json", 'w') as outfile:
    #     json.dump(result, outfile)
