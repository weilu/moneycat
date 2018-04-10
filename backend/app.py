from chalice import Chalice
import subprocess
import os
import io
import csv
from tempfile import NamedTemporaryFile
from chalicelib import pdftotxt
import cgi


app = Chalice(app_name='parseBankStatement')
app.api.binary_types.append('multipart/form-data')
app.debug = True

LAMBDA_TASK_ROOT = os.environ.get('LAMBDA_TASK_ROOT', os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(LAMBDA_TASK_ROOT, 'bin')
LIB_DIR = os.path.join(LAMBDA_TASK_ROOT, 'lib')


def get_multipart_data():
    content_type_obj = app.current_request.headers['content-type']
    content_type, property_dict = cgi.parse_header(content_type_obj)

    property_dict['boundary'] = bytes(property_dict['boundary'], "utf-8")
    body = io.BytesIO(app.current_request.raw_body)
    return cgi.parse_multipart(body, property_dict)


@app.route('/upload', methods=['POST'],
           content_types=['multipart/form-data'])
def upload():
    form_file = get_multipart_data()['file'][0]

    with NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
        f.write(form_file)
        filename = f.name

    output = io.StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerow(['date', 'description', 'amount', 'foreign_amount',
                         'statement_date', 'source'])
    pdftotxt.process_pdf(filename, csv_writer,
                pdftotxt_bin=os.path.join(BIN_DIR, 'pdftotext'),
                env=dict(LD_LIBRARY_PATH=LIB_DIR))

    # TODO: save pdf to s3
    # TODO: remove local tmp pdf
    return output.getvalue()


@app.route('/confirm', methods=['POST'],
           content_types=['multipart/form-data'])
def confirm():
    form_data = get_multipart_data()
    form_file = form_data['file'][0]
    uuid = form_data['uuid']
    # TODO: save csv to s3 by user
