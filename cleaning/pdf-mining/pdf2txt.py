#!/usr/bin/env python

"""
Converts PDF text content (though not images containing text) to plain text, html, xml or "tags".
"""
import io
import json

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
        line = line.strip().replace("\n", "")
        if re.search("REF NO :.", line):
            line = re.sub("REF NO :.*$", "", line)
            line = re.sub('\s+', " ", line)
            all_records.append(line)
        elif line is not "$":
            other_records.append(line)
    count_line = len(all_records)
    for i in range(count_line):
        result.append([other_records[i], all_records[i], other_records[count_line + i]])
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
    if idx == 1:
        begin = re.search('Description of Transaction', txt).span()[1]
        end = re.search('Contact Us', txt).span()[0]
        for line in txt[begin:end].split("\n\n"):
            line = line.strip().replace("\n", "")
            if len(line) == 0:
                continue
            if re.search("Ref No", line):
                line = re.sub("Ref No.*$", "", line)
                line = re.sub('\s+', " ", line)
            if re.search("^[0-9]{2} JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC$", line):
                data_record.append(line)
            else:
                trans_record.append(line)
        begin = re.search('Transaction Amount', txt).span()[1]
        for line in txt[begin:].split("\n\n"):
            line = line.strip().replace("\n", " ")
            if re.search("SGD", line) or len(line) == 0:
                continue
            # if re.search("Ref No", line):
            #     line = re.sub("Ref No.*$", "", line)
            #     line = re.sub('\s+', " ", line)
            count_record.append(line)
        # print(data_record, trans_record, count_record)
        # print(len(data_record), len(trans_record), len(count_record))
        # exit()
        date_idx = 0
        for i in range(len(trans_record)):
            if re.search("TOTAL|PREVIOUS|[^0-9a-zA-Z]+", trans_record[i]):
                continue
            result.append([data_record[2 * date_idx], trans_record[i], count_record[i]])
            date_idx += 1
    if idx == 2:
        begin = re.search('Description of Transaction', txt)
        if begin is None:
            return result
        else:
            begin = begin.span()[1]
        for line in txt[begin:].split("\n\n"):
            line = line.strip().replace("\n", " ")
            if re.search("SGD|^UOB.*$|"
                         "LU WEI|PostDate|TransDate|Description of Transaction|"
                         "Page [0-9]+ of [0-9]+", line) or len(line) == 0:
                continue
            if re.search("^[0-9]{2} JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC$", line) and len(line) == 6:
                data_record.append(line)
            elif re.search("^([+-]?)((\d{1,3}(,\d{3})*)|(\d+))(\.\d{2})?(\sCR)*$", line):
                count_record.append(line)
            else:
                trans_record.append(line)
        date_idx = 0
        for i in range(len(trans_record)):
            if re.search("TOTAL|PREVIOUS|[^0-9a-zA-Z]+", trans_record[i]):
                continue
            result.append([data_record[2 * date_idx], trans_record[i], count_record[i]])
            date_idx += 1
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
    # uob
    dir = "assets/uob"
    for filename in os.listdir(dir):
        if filename.endswith(".pdf"):
            print(filename)
            txt = pdf2txt(dir + "/" + filename)
            result += uob_parse(txt)
    with open(dir + "/out.json", 'w') as outfile:
        json.dump(result, outfile)
