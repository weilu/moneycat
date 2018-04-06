from chalice import Chalice
import subprocess
import os
import io
import csv
from tempfile import NamedTemporaryFile
from pdftotxt import process_pdf
import cgi


app = Chalice(app_name='parseBankStatement')
app.debug = True

LAMBDA_TASK_ROOT = os.environ.get('LAMBDA_TASK_ROOT', os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(LAMBDA_TASK_ROOT, 'bin')
LIB_DIR = os.path.join(LAMBDA_TASK_ROOT, 'lib')


@app.route('/upload', methods=['POST'],
           content_types=['multipart/form-data'])
def upload():
    content_type_obj = app.current_request.headers['content-type']
    content_type, property_dict = cgi.parse_header(content_type_obj)

    property_dict['boundary'] = bytes(property_dict['boundary'], "utf-8")
    body = io.BytesIO(app.current_request.raw_body)
    form_data = cgi.parse_multipart(body, property_dict)
    form_file = form_data['file'][0]

    with NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
        f.write(form_file)

    tmp_out = '/tmp/out_wei.csv'
    with open(tmp_out, 'w', encoding='utf-8') as o:
        csv_writer = csv.writer(o)
        csv_writer.writerow(['date', 'description', 'amount', 'foreign_amount',
                             'statement_date', 'source'])
        process_pdf(f.name, csv_writer,
                    pdftotxt_bin=os.path.join(BIN_DIR, 'pdftotext'),
                    env=dict(LD_LIBRARY_PATH=LIB_DIR))

    return open(tmp_out).read()

