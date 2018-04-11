from chalice import Chalice
import subprocess
import os
import io
import csv
from tempfile import NamedTemporaryFile
from chalicelib import pdftotxt
import cgi
import boto3
import logging
from sklearn.externals import joblib
import pandas as pd


app = Chalice(app_name='parseBankStatement')
app.api.binary_types.append('multipart/form-data')
app.debug = True
app.log.setLevel(logging.DEBUG)

lambda_task_root = os.environ.get('LAMBDA_TASK_ROOT',
                                  os.path.dirname(os.path.abspath(__file__)))
# Only exists in production lambda env
maybe_exist = os.path.join(lambda_task_root, 'pdftotext')
if os.path.isdir(maybe_exist):
    lambda_task_root = maybe_exist
BIN_DIR = os.path.join(lambda_task_root, 'bin')
LIB_DIR = os.path.join(lambda_task_root, 'lib')

PDF_BUCKET = 'cs4225-bank-pdfs'
MODEL_BUCKET = 'cs4225-models'
s3 = boto3.client('s3')

def get_multipart_data():
    content_type_obj = app.current_request.headers['content-type']
    content_type, property_dict = cgi.parse_header(content_type_obj)

    property_dict['boundary'] = bytes(property_dict['boundary'], "utf-8")
    body = io.BytesIO(app.current_request.raw_body)
    return cgi.parse_multipart(body, property_dict)


def get_model(name):
    with io.BytesIO() as f:
        s3.download_fileobj(MODEL_BUCKET, name, f)
        f.seek(0)
        return joblib.load(f)

@app.route('/upload', methods=['POST'], content_types=['multipart/form-data'])
def upload():
    form_file = get_multipart_data()['file'][0]

    with NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
        filename = f.name
        f.write(form_file)

    # upload to s3
    key_name = os.path.basename(filename)
    app.log.debug('uploading {} to s3'.format(key_name))
    s3.upload_file(filename, PDF_BUCKET, key_name)

    # parse
    # TODO: parsing is slow, need to improve performance
    output = io.StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerow(['date', 'description', 'amount', 'foreign_amount',
                         'statement_date', 'source'])
    app.log.debug('start parsing pdf')
    pdftotxt.process_pdf(filename, csv_writer,
                pdftotxt_bin=os.path.join(BIN_DIR, 'pdftotext'),
                env=dict(LD_LIBRARY_PATH=LIB_DIR))
    app.log.debug('done parsing pdf')
    os.remove(filename)

    # classification
    classifier = get_model("svm_classifier.pkl")
    transformer = get_model("tfidf_transformer.pkl")
    label_transformer = get_model("label_transformer.pkl")
    output.seek(0)
    df = pd.read_csv(output)
    pred = classifier.predict(transformer.transform(df['description']))
    categories = label_transformer.inverse_transform(pred)
    df['category'] = categories

    return df.to_csv()


@app.route('/confirm', methods=['POST'], content_types=['multipart/form-data'])
def confirm():
    form_data = get_multipart_data()
    form_file = form_data['file'][0]
    uuid = form_data['uuid']
    # TODO: save csv to s3 by user

