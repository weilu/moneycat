from chalice import Chalice, CORSConfig, Response, CognitoUserPoolAuthorizer
import subprocess
import os
import io
import csv
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs
from chalicelib import pdftotxt
from chalicelib.algo import reservior_sampling
from chalicelib.train import export_model
import cgi
import boto3
import logging
from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
from dateparser.search import search_dates
from datetime import datetime
import hashlib

# TODO: stricter cors rules
# cors_config = CORSConfig(
#     allow_origin='https://weilu.github.io/cs4225',
# )

app = Chalice(app_name='parseBankStatement')
app.api.binary_types.append('multipart/form-data')
app.debug = True
app.log.setLevel(logging.DEBUG)

ENV = os.environ.get('ENV', 'dev')
PDF_BUCKET = 'moneycat-pdfs-{}'.format(ENV)
REQUEST_PDF_BUCKET = 'moneycat-request-pdfs-{}'.format(ENV)
DB_NAME = 'moneycat-{}'.format(ENV)

if ENV == 'dev':
    lambda_task_root = '/usr/local/'
else:
    lambda_task_root = os.path.dirname(os.path.abspath(__file__))
# Only exists in non-local lambda env
maybe_exist = os.path.join(lambda_task_root, 'pdftotext')
if os.path.isdir(maybe_exist):
    lambda_task_root = maybe_exist
BIN_DIR = os.path.join(lambda_task_root, 'bin')
LIB_DIR = os.path.join(lambda_task_root, 'lib')

MODEL_BUCKET = 'cs4225-models'
CLASSIFIER_FILENAME = "svm_classifier.pkl"
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


def dynamodb_response_to_df(response):
    df = pd.DataFrame.from_dict(response['Items'])
    if not df.empty:
        df.drop(columns=['txid', 'uuid', 'updated_at'], inplace=True)

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
        payload = df.to_csv()
        content_type = 'text/csv'

    return Response(body=payload, headers={'Content-Type': content_type})


@app.route('/upload', methods=['POST'],
           content_types=['multipart/form-data'], cors=True,
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
        if not pd.isnull(row['foreign_amount']):
            item["foreign_amount"] = { "S": row['foreign_amount'] }

        requests.append({"PutRequest": { "Item": item}})
        if len(requests) == 25:
            send_write_request(requests)
            requests = [] # reset requests buffer for next batch
    if len(requests) > 0:
        send_write_request(requests)


@app.route('/confirm', methods=['POST'],
           content_types=['application/x-www-form-urlencoded'], cors=True,
           authorizer=get_authorizer())
def confirm():
    form_data = parse_qs(app.current_request.raw_body.decode())
    if 'file' not in form_data:
        return Response(body='file must be present', status_code=400)
    form_file = form_data['file'][0]
    uuid = get_current_user_email()
    if not form_file:
        return Response(body='Invalid file {}'.format(form_file),
                        status_code=400)

    # update dynamoDB
    file_io = io.StringIO(form_file)
    accept_header = app.current_request.headers.get('accept')
    if accept_header and 'application/json' in accept_header:
        df = pd.read_json(file_io, orient='records', convert_dates=False)
    else:
        df = pd.read_csv(file_io, index_col=False)
    batch_tx_writes(uuid, df)

    return Response(body='', status_code=201)


@app.route('/update', methods=['POST'],
           content_types=['application/x-www-form-urlencoded'], cors=True,
           authorizer=get_authorizer())
def update():
    form_data = parse_qs(app.current_request.raw_body.decode())
    if 'description' not in form_data or 'category' not in form_data:
        return Response(body='Required form fields: description and category must be present',
                        status_code=400)
    description = form_data['description'][0]
    category = form_data['category'][0]
    if not description or not category:
        return Response(body='Invalid description {} or category {}'\
                .format(description, category), status_code=400)

    # remove dates from description to maximize description matching
    search_result = search_dates(description, languages=['en'])
    if search_result:
        date_strings = [pair[0] for pair in search_result]
        for date in date_strings:
            description = description.replace(date, '')
    # TODO validate category, later

    uuid = get_current_user_email()
    query_params = query_by_uuid_param(uuid)
    query_params['FilterExpression'] = 'contains(description, :des_value)'
    query_params['ExpressionAttributeValues'][':des_value'] = {'S': description}
    response = dynamodb.query(**query_params)
    items = response['Items']
    updated_at = str(datetime.utcnow())
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
        print(update_response) # TODO handle failure & partial success cases

    return Response(body='Updated {} transactions'.format(len(items)), status_code=200)


@app.route('/transactions', methods=['GET'], cors=True, authorizer=get_authorizer())
def transactions():
    uuid = get_current_user_email()
    response = dynamodb.query(**query_by_uuid_param(uuid))
    df = dynamodb_response_to_df(response)
    return dataframe_as_response(df, app.current_request.headers.get('accept'))


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


@app.route('/request', methods=['POST'],
           content_types=['multipart/form-data'], cors=True,
           authorizer=get_authorizer())
def request():
    form_data = get_multipart_data()
    form_file = form_data['file'][0]
    password = form_data['password'][0] if 'password' in form_data else ''

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


def get_current_user_email():
    req_context = app.current_request.context
    return req_context['authorizer']['claims']['email']
