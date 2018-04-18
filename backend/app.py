from chalice import Chalice, CORSConfig, Response
import subprocess
import os
import io
import csv
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs
from chalicelib import pdftotxt
import cgi
import boto3
import logging
from sklearn.externals import joblib
import pandas as pd

import random


# TODO: stricter cors rules
# cors_config = CORSConfig(
#     allow_origin='https://weilu.github.io/cs4225',
# )

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
CSV_BUCKET = 'cs4225-bank-csvs'
MODEL_BUCKET = 'cs4225-models'
CLASSIFIER_FILENAME = "svm_classifier.pkl"
s3 = boto3.client('s3')


def get_multipart_data():
    content_type_obj = app.current_request.headers['content-type']
    content_type, property_dict = cgi.parse_header(content_type_obj)

    property_dict['boundary'] = bytes(property_dict['boundary'], "utf-8")
    body = io.BytesIO(app.current_request.raw_body)
    return cgi.parse_multipart(body, property_dict)


def get_model(name):
    with io.BytesIO() as f:
        model_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=name)
        f.write(model_obj['Body'].read())
        f.seek(0)
        return (joblib.load(f), model_obj)


def s3_csvs_to_df(files):
    df = pd.DataFrame()
    for file_meta in files:
        print(file_meta['Key'])
        csv_file = s3.get_object(Bucket=CSV_BUCKET, Key=file_meta['Key'])
        csv_io = io.StringIO(csv_file['Body'].read().decode('utf-8'))
        df = df.append(pd.read_csv(csv_io), ignore_index=True)
        df.drop_duplicates(inplace=True)
    return df


# new_data and existing_samples should be of type pd.DataFrame
def reservior_sampling(sample_size, new_data,
                       existing_record_count=0, existing_samples=pd.DataFrame()):
    new_data = new_data.reset_index(drop=True)
    samples = existing_samples.reset_index(drop=True)
    replaced_indexes = {}
    for index, record in new_data.iterrows():
        if samples.shape[0] < sample_size:
            samples = samples.append(record, ignore_index=True)
            replaced_indexes[index] = index
        else:
            r = random.randint(0, existing_record_count + index)
            if r < sample_size:
                samples.loc[r] = record
                replaced_indexes[r] = index
    samples.reset_index(inplace=True)
    remaining_indexes = set(range(len(new_data))) - set(replaced_indexes.values())
    return (samples, remaining_indexes)


@app.route('/upload', methods=['POST'],
           content_types=['multipart/form-data'], cors=True)
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
                         'statement_date'])
    app.log.debug('start parsing pdf')
    pdftotxt.process_pdf(filename, csv_writer,
                pdftotxt_bin=os.path.join(BIN_DIR, 'pdftotext'),
                include_source=False,
                env=dict(LD_LIBRARY_PATH=LIB_DIR))
    app.log.debug('done parsing pdf')
    os.remove(filename)

    # classification
    classifier = get_model(CLASSIFIER_FILENAME)[0]
    transformer = get_model("tfidf_transformer.pkl")[0]
    label_transformer = get_model("label_transformer.pkl")[0]
    output.seek(0)
    df = pd.read_csv(output)
    pred = classifier.predict(transformer.transform(df['description']))
    categories = label_transformer.inverse_transform(pred)
    df['category'] = categories

    return {'csv': df.to_csv()}


@app.route('/confirm', methods=['POST'],
           content_types=['application/x-www-form-urlencoded'], cors=True)
def confirm():
    form_data = parse_qs(app.current_request.raw_body.decode())
    if 'file' not in form_data or 'uuid' not in form_data:
        return Response(body='Both file and uuid must be present',
                        status_code=400)
    form_file = form_data['file'][0]
    uuid = form_data['uuid'][0]
    if not uuid or not form_file:
        return Response(body='Invalid uuid {} or file {}'.format(uuid, form_file),
                        status_code=400)

    with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filename = f.name
        f.write(form_file)

    # upload to s3
    key_name = "{}/{}".format(uuid, os.path.basename(filename))
    app.log.debug('uploading {} to s3'.format(key_name))
    s3.upload_file(filename, CSV_BUCKET, key_name)
    os.remove(filename)

    return Response(body='', status_code=201)


@app.route('/transactions/{uuid}', methods=['GET'], cors=True)
def transactions(uuid):
    files = s3.list_objects(Bucket=CSV_BUCKET, Prefix=uuid)['Contents']
    df = s3_csvs_to_df(files)
    return df.to_csv()


@app.route('/refresh-model', methods=['GET'])
def refresh_model():
    classifier, model_obj = get_model(CLASSIFIER_FILENAME)
    get_last_modified = lambda obj: obj['LastModified']
    compare_last_modified = lambda obj: get_last_modified(obj) >= last_refresh_ts
    last_refresh_ts = get_last_modified(model_obj)
    files = s3.list_objects(Bucket=CSV_BUCKET)['Contents']

    # get all CSVs created since last refresh
    sorted_files = sorted(filter(compare_last_modified, files),
                          key=get_last_modified, reverse=True)

    # make sure we don't load more than what we can handle in memory
    size = 0
    file_metas = []
    for obj in sorted_files:
        size += obj['Size']
        if size > 1000000000: # 1GB
            app.log.warning('Unprocessed files greater than 1GB! Processing the latest 1GB')
            break
        app.log.debug('Updating model with file: {} {} {}'.format(obj['Key'], obj['Size'], obj['LastModified']))
        file_metas.append(obj)

    if not file_metas: # no new data to update model with
        return Response(body='', status_code=304)

    df = s3_csvs_to_df(file_metas)

    label_transformer = get_model("label_transformer.pkl")[0]
    y = label_transformer.transform(df['category'])

    transformer = get_model("tfidf_transformer.pkl")[0]
    X = df['description']
    X_train = transformer.transform(X)

    # update model
    classifier = classifier.partial_fit(X_train, y)

    # serialize & upload to s3
    # TODO: only save classifier if accuracy is higher
    with NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        filename = f.name
    joblib.dump(classifier, filename)
    s3.upload_file(filename, MODEL_BUCKET, CLASSIFIER_FILENAME)
