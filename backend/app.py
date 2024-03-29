from chalice import Chalice, CORSConfig, Response, CognitoUserPoolAuthorizer
import subprocess
import os
import io
import csv
from tempfile import NamedTemporaryFile
from chalicelib import pdftotxt
from chalicelib.algo import reservior_sampling
from chalicelib.train import export_model
from chalicelib.active import get_subcategory_to_category_map
import cgi
import boto3
from botocore.exceptions import ClientError
import logging
from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
from dateparser.search import search_dates
from datetime import datetime
import hashlib
import re
from json.decoder import JSONDecodeError
import json

ORIGINS = {'dev': '*', 'prod': 'https://moneycat.sg'}

app = Chalice(app_name='parseBankStatement')
app.api.binary_types.append('multipart/form-data')
app.debug = True
app.log.setLevel(logging.DEBUG)

ENV = os.environ.get('ENV', 'dev')
PDF_BUCKET = 'moneycat-pdfs-{}'.format(ENV)
REQUEST_PDF_BUCKET = 'moneycat-request-pdfs-{}'.format(ENV)
DB_NAME = 'moneycat-{}'.format(ENV)
CORS_CONFIG = CORSConfig(
    allow_origin=ORIGINS[ENV],
    allow_credentials=True
)

local_lambda_task_root = '/usr/local/' # for osx
travis_binary_root = '/usr/' # for travis CI only
if os.path.exists(local_lambda_task_root + 'bin/pdftotext'):
    lambda_task_root = local_lambda_task_root
elif os.path.exists(travis_binary_root + 'bin/pdftotext'):
    lambda_task_root = travis_binary_root
else:
    lambda_task_root = os.path.dirname(os.path.abspath(__file__))
# Only exists in non-local lambda env
maybe_exist = os.path.join(lambda_task_root, 'pdftotext')
if os.path.isdir(maybe_exist):
    lambda_task_root = maybe_exist
BIN_DIR = os.path.join(lambda_task_root, 'bin')
LIB_DIR = os.path.join(lambda_task_root, 'lib')

MODEL_BUCKET = 'cs4225-models'
CLASSIFIER_FILENAME = "classifier.pkl"
META_FILENAME = 'meta.pkl'

s3 = boto3.client('s3')
dynamodb = boto3.client('dynamodb')
authorizer = None


def get_authorizer():
    global authorizer
    if not authorizer:
        authorizer = CognitoUserPoolAuthorizer(os.environ.get('AUTH_POOL_NAME'),
            provider_arns=[os.environ.get('AUTH_ARN')])
    return authorizer


def query_by_uuid_param(uuid):
    return {'TableName': DB_NAME,
            'ExpressionAttributeNames': {'#uuid': 'uuid'},
            'KeyConditionExpression': '#uuid = :uuid_val',
            'ExpressionAttributeValues': {':uuid_val': {'S': uuid}}}


def query_by_uuid_and_txid_param(uuid, txid):
    return {
            'TableName': DB_NAME,
            'Key': {'uuid': {'S': uuid}, 'txid': {'S': txid}},
            'ExpressionAttributeNames': {'#uuid': 'uuid'},
            'ConditionExpression': "#uuid = :uuid_val AND txid = :txid_val",
            'ExpressionAttributeValues': {
                ':uuid_val': {'S': uuid}, ':txid_val': {'S': txid}
            }
           }


def get_multipart_data():
    content_type_obj = app.current_request.headers['content-type']
    content_type, property_dict = cgi.parse_header(content_type_obj)

    if 'boundary' not in property_dict:
        return None

    property_dict['boundary'] = bytes(property_dict['boundary'], "utf-8")
    body = io.BytesIO(app.current_request.raw_body)
    return cgi.parse_multipart(body, property_dict)


def get_model(name):
    with io.BytesIO() as f:
        model_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=name)
        f.write(model_obj['Body'].read())
        f.seek(0)
        return (joblib.load(f), model_obj)


def get_current_user_email():
    req_context = app.current_request.context
    if ENV == 'dev':
        return 'wei' # fixture user for test & local dev purposes
    return req_context['authorizer']['claims']['email']


def dynamodb_response_to_df(response, include_txid=False):
    df = pd.DataFrame.from_dict(response['Items'])
    if not df.empty:
        df.drop(columns=['uuid', 'updated_at'], inplace=True)
        if not include_txid:
            df.drop(columns=['txid'], inplace=True)

        def convert_dynamo_data_type(type_value):
            if pd.isnull(type_value):
                return type_value
            key = list(type_value.keys())[0]
            value = type_value[key]
            if 'N' == key:
                return float(value)
            else:
                return value
        df = df.applymap(convert_dynamo_data_type)
    return df


def dataframe_as_response(df, accept_header):
    if accept_header and 'application/json' in accept_header:
        payload = df.to_json(orient='records')
        content_type = 'application/json'
    else: # default to csv on unknown format
        payload = df.to_csv(index=False)
        content_type = 'text/csv'

    return Response(body=payload, headers={'Content-Type': content_type})


@app.route('/upload', methods=['POST'], cors=CORS_CONFIG,
           content_types=['multipart/form-data'],
           authorizer=get_authorizer())
def upload():
    form_data = get_multipart_data()
    if not form_data:
        return Response(body='Missing form data', status_code=400)

    if 'file' not in form_data or not form_data['file'] or not form_data['file'][0]:
        return Response(body='Missing upload file', status_code=400)
    form_file = form_data['file'][0]

    password = form_data['password'][0] if 'password' in form_data else None

    with NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
        filename = f.name
        f.write(form_file)

    # upload to s3
    key_name = os.path.basename(filename)
    app.log.debug('uploading {} to s3'.format(key_name))
    s3.upload_file(filename, PDF_BUCKET, key_name)

    # parse
    output = io.StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerow(['date', 'description', 'amount', 'foreign_amount',
                         'statement_date'])
    app.log.debug('start parsing pdf')
    try:
        pdftotxt.process_pdf(filename, csv_writer,
                    pdftotxt_bin=os.path.join(BIN_DIR, 'pdftotext'),
                    include_source=False,
                    password=password,
                    env=dict(LD_LIBRARY_PATH=LIB_DIR))
    except RuntimeError as e:
        return Response(body=e.args[0], status_code=400)
    finally:
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

    return dataframe_as_response(df, app.current_request.headers.get('accept'))


def send_write_request(requests):
    request = {DB_NAME: requests}
    response = dynamodb.batch_write_item(RequestItems=request,
            ReturnConsumedCapacity='TOTAL')
    print(response)


def batch_tx_writes(uuid, tx_df):
    # ignore category as the same transaction may be classified differently by different models
    content = tx_df.drop(columns=['category']).to_csv()
    content_hash = hashlib.sha1(content.encode('utf-8')).hexdigest()
    updated_at = str(datetime.utcnow())
    requests = []
    for index, row in tx_df.iterrows():
        item = {
          "uuid": {
            "S": uuid
          },
          "txid": {
            "S": row['date'] + '-' + content_hash + '-' + format(index, '04')
          },
          "date": {
            "S": row['date']
          },
          "description": {
            "S": row['description']
          },
          "amount": {
            "N": str(row['amount'])
          },
          "statement_date": {
            "S": row['statement_date']
          },
          "category": {
            "S": row['category']
          },
          "updated_at": {
            "S": updated_at
          },
        }
        if 'foreign_amount' in row and not pd.isnull(row['foreign_amount']):
            item["foreign_amount"] = { "S": row['foreign_amount'] }

        requests.append({"PutRequest": { "Item": item}})
        if len(requests) == 25:
            send_write_request(requests)
            requests = [] # reset requests buffer for next batch
    if len(requests) > 0:
        send_write_request(requests)


@app.route('/confirm', methods=['POST'], cors=CORS_CONFIG,
           content_types=['application/json', 'text/csv'],
           authorizer=get_authorizer())
def confirm():
    body = app.current_request.raw_body.decode()
    if app.current_request.json_body:
        df = pd.read_json(io.StringIO(body), orient='records',
                          convert_dates=False)
    elif body:
        df = pd.read_csv(io.StringIO(body), index_col=False)
    else:
        df = pd.DataFrame()

    if df.empty:
        msg = ("Missing or invalid request payload. Make sure it's in"
               "json or csv format with a matching Content-Type header.")
        return Response(body=msg, status_code=400)

    uuid = get_current_user_email()
    batch_tx_writes(uuid, df)

    return Response(body='', status_code=201)


@app.route('/update', methods=['POST'], content_types=['application/json'],
           cors=CORS_CONFIG, authorizer=get_authorizer())
def update():
    try:
        form_data = app.current_request.json_body
    except JSONDecodeError:
        return Response(body='Malformed JSON', status_code=400)

    if 'description' not in form_data or 'category' not in form_data:
        return Response(body='Required form fields: description and category must be present',
                        status_code=400)
    description = form_data['description']
    category = form_data['category']
    if not description or not category:
        return Response(body='Invalid description {} or category {}'\
                .format(description, category), status_code=400)

    ### remove known noise from description to maximize description matching ###
    # remove dollar amount e.g. $110.12, sometimes it's misinterpreted by dateparser
    description = re.sub(r'([$]\d+(?:\.\d{2})?)', '', description)
    # remove auto-split transactions e.g. 001/003 in case of OCBC
    description = re.sub(r'(\d{3}/\d{3})', '', description)
    # remove dates
    search_result = search_dates(description, languages=['en'])
    if search_result:
        date_strings = [pair[0] for pair in search_result]
        for date in date_strings:
            description = description.replace(date, '')
    description = description.strip()

    # TODO validate category, later

    uuid = get_current_user_email()
    query_params = query_by_uuid_param(uuid)
    query_params['FilterExpression'] = 'contains(description, :des_value)'
    query_params['ExpressionAttributeValues'][':des_value'] = {'S': description}
    response = dynamodb.query(**query_params)
    items = response['Items']
    updated_at = str(datetime.utcnow())
    updated_tx_ids = set()
    for tx in items:
        key_params = {k: v for k, v in tx.items() if k in ['uuid', 'txid']}
        update_params = {'TableName': DB_NAME,
                'Key': key_params,
                'UpdateExpression': 'SET category = :new_cat_value, updated_at = :updated_at',
                'ExpressionAttributeValues': {
                    ':new_cat_value': {'S': category},
                    ':updated_at': {'S': updated_at}
                },
                'ReturnValues': 'UPDATED_NEW'}
        update_response = dynamodb.update_item(**update_params)
        if update_response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:
            updated_tx_ids.add(key_params['txid']['S'])
        else:
            print(update_response) # TODO handle failure & partial success cases

    return sorted(list(updated_tx_ids))


@app.route('/transactions', methods=['GET'], cors=CORS_CONFIG,
           authorizer=get_authorizer())
def transactions():
    uuid = get_current_user_email()
    response = dynamodb.query(**query_by_uuid_param(uuid))
    query_params = app.current_request.query_params
    include_txid = query_params and query_params.get('txid')
    df = dynamodb_response_to_df(response, include_txid)
    return dataframe_as_response(df, app.current_request.headers.get('accept'))


@app.route('/transactions/{txid}', methods=['DELETE'], cors=CORS_CONFIG,
           authorizer=get_authorizer())
def delete_transactions(txid):
    uuid = get_current_user_email()
    try:
        dynamodb.delete_item(**query_by_uuid_and_txid_param(uuid, txid))
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return Response(body=f'Invalid transaction id: {txid}',
                            status_code=400)
    return Response(body=f'Deleted {txid}', status_code=200)

@app.route('/refresh-model', methods=['GET'])
def refresh_model():
    classifier, model_obj = get_model(CLASSIFIER_FILENAME)
    # note: LastModified is based on when the model file is last written on s3
    # using it as query condition may miss transactions written to db while
    # this function is running. It's not critical.
    get_last_modified = lambda obj: obj['LastModified']
    last_refresh_ts = str(get_last_modified(model_obj))
    last_refresh_ts = last_refresh_ts[0:last_refresh_ts.index('+')]
    response = dynamodb.scan(TableName=DB_NAME,
      FilterExpression='updated_at >= :updated_at_val',
      ExpressionAttributeValues={':updated_at_val': {'S': last_refresh_ts}})
    df = dynamodb_response_to_df(response)

    if df.empty: # no new data to update model with
        return Response(body='No new data to update model with', status_code=200)

    label_transformer = get_model("label_transformer.pkl")[0]
    y = label_transformer.transform(df['category'])

    X = df['description']

    existing_test_samples = get_model('test_samples.pkl')[0]
    meta_data = get_model(META_FILENAME)[0]
    new_df = pd.DataFrame(data={'X': X, 'y': y})
    samples, remaining_ids = reservior_sampling(50, new_df,
            meta_data['train_size'], existing_test_samples)

    transformer = get_model("tfidf_transformer.pkl")[0]
    X_test = transformer.transform(samples['X'])
    y_test = samples['y']
    score_before = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print('Accuracy before update: {}'.format(score_before))

    # Test prediction with updated model
    X_train = transformer.transform(new_df['X'][remaining_ids])
    y_train = new_df['y'][remaining_ids]
    classifier = classifier.partial_fit(X_train, y_train)
    score_after = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print('Accuracy after update: {}, diff: {}'\
          .format(score_after, score_after - score_before))

    # did not improve model, do not update
    if score_after <= score_before:
        msg = 'Accuracy with additional data is {}, which is no better than previous accuracy {}'\
                .format(score_after, score_before)
        return Response(body=msg, status_code=200)

    # serialize & upload everything to s3
    pred = classifier.predict(X_test)
    report = metrics.classification_report(y_test, pred,
            target_names=list(label_transformer.classes_))
    print(report)
    meta_data = {'train_size': meta_data['train_size'] + X_train.shape[0],
                 'accuracy': score_after}
    export_model(classifier, transformer, label_transformer, samples, meta_data, report)

    msg = 'New model improved accuracy from {} to {}'.format(score_before, score_after)
    return Response(body=msg, status_code=201)


@app.route('/request', methods=['POST'], content_types=['multipart/form-data'],
           cors=CORS_CONFIG, authorizer=get_authorizer())
def request():
    form_data = get_multipart_data()
    form_file = form_data['file'][0]
    password = form_data['password'][0] if 'password' in form_data else ''
    if hasattr(password, 'decode'): # in case password somehow comes in as binary data
        password = password.decode()

    with NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
        filename = f.name
        f.write(form_file)

    # upload to s3
    key_name = os.path.basename(filename)
    app.log.debug('uploading {} to s3'.format(key_name))
    s3.upload_file(filename, REQUEST_PDF_BUCKET, key_name)

    # tag file with user email and password
    tag_args = {'TagSet': [
        {'Key': 'uuid', 'Value': get_current_user_email()},
        {'Key': 'password', 'Value': password}]}
    response = s3.put_object_tagging(Bucket=REQUEST_PDF_BUCKET,
                                     Key=key_name, Tagging=tag_args)
    app.log.debug(response)

    return Response(body='', status_code=201)


@app.route('/categories', methods=['GET'], cors=CORS_CONFIG)
def categories():
    all_subcats = get_subcategory_to_category_map()
    body = json.dumps(all_subcats, indent=2, sort_keys=True)
    etag = hashlib.md5(body.encode('utf-8')).hexdigest()

    if_none_match = app.current_request.headers.get('if-none-match')
    if if_none_match == etag:
        return Response(body='', status_code=304)

    return Response(body=body,
                    headers={'Content-Type': 'application/json',
                             'ETag': etag})


@app.route('/testauth', methods=['GET'], cors=CORS_CONFIG,
           authorizer=get_authorizer())
def testauth():
    return {"success": True}
