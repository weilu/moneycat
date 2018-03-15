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
    # print(txt)
    # exit()
    result = []
    tmp_txt = ""
    tmp_txt2 = ""
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
    dir = "assets"
    for filename in os.listdir(dir):
        if filename.endswith(".pdf"):
            txt = pdf2txt(dir + "/" + filename)
            result = dbs_parse(txt)
            with open(dir + "/" + filename.replace(".pdf", "") + ".json", 'w') as outfile:
                json.dump(result, outfile)
