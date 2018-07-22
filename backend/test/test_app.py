import unittest
import os
import io
import base64
import pandas as pd
import json
from chalice.config import Config
from chalice.local import LocalGateway
from app import app, s3, send_write_request, dynamodb
from requests_toolbelt import MultipartEncoder
from pandas.util.testing import assert_frame_equal
import hashlib

class TestApp(unittest.TestCase):

    def setUp(self):
        self.lg = LocalGateway(app, Config())

    def test_upload(self):
        payload = self.get_pdf_payload()
        response = self.lg.handle_request(method='POST', path='/upload',
                headers={'Content-Type': payload.content_type}, body=payload.to_string())

        self.assertEqual(response['statusCode'], 200)
        expected_csv = os.path.join(os.path.dirname(__file__), 'data', 'uob.csv')
        self.assertEqual(response['body'], self.read_and_close(expected_csv))
        self.check_and_cleanup_pdf()

    def test_upload_json(self):
        payload = self.get_pdf_payload()
        response = self.lg.handle_request(method='POST', path='/upload',
                headers={'Content-Type': payload.content_type,
                         'Accept': 'application/json'},
                body=payload.to_string())

        self.assertEqual(response['statusCode'], 200)
        expected_json = os.path.join(os.path.dirname(__file__), 'data', 'uob.json')
        self.assertEqual(response['body'], self.read_and_close(expected_json).strip())
        self.check_and_cleanup_pdf()

    def test_upload_with_password(self):
        payload = self.get_pdf_payload('123abc')
        response = self.lg.handle_request(method='POST', path='/upload',
                headers={'Content-Type': payload.content_type}, body=payload.to_string())

        self.assertEqual(response['statusCode'], 200)
        expected_csv = os.path.join(os.path.dirname(__file__), 'data', 'uob.csv')
        self.assertEqual(response['body'], self.read_and_close(expected_csv))
        self.check_and_cleanup_pdf()

    def test_upload_bad_request(self):
        response = self.lg.handle_request(method='POST', path='/upload',
                headers={'Content-Type': 'multipart/form-data'}, body='')
        self.assertEqual(response['statusCode'], 400)
        self.assertEqual(response['body'], 'Missing form data')

        payload = MultipartEncoder({'password': '123'})
        response = self.lg.handle_request(method='POST', path='/upload',
                headers={'Content-Type': payload.content_type}, body=payload.to_string())
        self.assertEqual(response['statusCode'], 400)
        self.assertEqual(response['body'], 'Missing upload file')

    def test_confirm_and_transactions(self):
        get_headers = {}
        post_headers = {'Content-Type': 'text/csv'}
        parse_fn = lambda body: pd.read_csv(body, index_col=False)
        self.verify_confirm_and_transactions(get_headers, post_headers, 'uob.csv', parse_fn)

    def test_confirm_and_transactions_json(self):
        get_headers = {'Accept': 'application/json'}
        post_headers = {'Content-Type': 'application/json'}
        parse_fn = lambda body: pd.read_json(body,
                orient='records', convert_dates=False)
        self.verify_confirm_and_transactions(get_headers, post_headers, 'uob.json', parse_fn)

    def verify_confirm_and_transactions(self,
            get_request_headers, post_request_headers,
            request_payload_filename, response_body_parse_fn):
        self.delete_all_tx_of_test_user(get_request_headers, response_body_parse_fn)

        # create new transactions
        filename = os.path.join(os.path.dirname(__file__), 'data',
                request_payload_filename)
        payload = self.read_and_close(filename)
        response = self.lg.handle_request(method='POST', path='/confirm',
                headers=post_request_headers, body=payload)
        self.assertEqual(response['statusCode'], 201)

        # verify transactions created
        response = self.lg.handle_request(method='GET', path='/transactions',
                headers=get_request_headers, body='')
        self.assertEqual(response['statusCode'], 200)
        self.assert_str_as_dataframe_equal(io.StringIO(response['body']), filename,
                                        response_body_parse_fn)
    def test_confirm_invalid_payload(self):
        # missing payload
        response = self.lg.handle_request(method='POST', path='/confirm',
                headers={}, body='')
        self.assertEqual(response['statusCode'], 400)

        # mismatched content type
        filename = os.path.join(os.path.dirname(__file__), 'data', 'uob.json')
        payload = self.read_and_close(filename)
        response = self.lg.handle_request(method='POST', path='/confirm',
                headers={'Content-Type': 'text/csv'}, body=payload)
        self.assertEqual(response['statusCode'], 400)

    def test_update(self):
        self.delete_all_tx_of_test_user()

        confirm_payload = '''
date,description,amount,statement_date,category
2016-07-13,10 JUL CR INTEREST,-40.59,2016-08-12,Interest Income
2016-07-18,15 JUL CR INTEREST,-72.97,2016-08-12,Interest Income
2016-07-15,13 JUL GRAIN - GO106604 SINGAPORE,45.75,2016-08-12,Delivery
2016-07-18,CGH CLINICS $110.12 001/003,75,2016-08-12,Dentist
2016-07-18,CGH CLINICS $110.12 002/003,75,2016-08-12,Dentist
2016-07-18,CGH CLINICS $110.12 003/003,75,2016-08-12,Dentist
'''

        response = self.lg.handle_request(method='POST', path='/confirm',
                headers={'Content-type': 'text/csv'}, body=confirm_payload)
        self.assertEqual(response['statusCode'], 201)

        update_payload = json.dumps({
            'description': '10 JUL CR INTEREST',
            'category': 'Returned Purchase'})

        response = self.lg.handle_request(method='POST', path='/update',
                headers={'Content-type': 'application/json'}, body=update_payload)
        self.assertEqual(response['statusCode'], 200)
        self.assertEqual(response['body'], 'Updated 2 transactions')

        update_payload = json.dumps({
            'description': 'CGH CLINICS $110.12 003/003',
            'category': 'Doctor'})

        response = self.lg.handle_request(method='POST', path='/update',
                headers={'Content-type': 'application/json'}, body=update_payload)
        self.assertEqual(response['statusCode'], 200)
        self.assertEqual(response['body'], 'Updated 3 transactions')

        expected_payload = '''
date,description,amount,statement_date,category
2016-07-13,10 JUL CR INTEREST,-40.59,2016-08-12,Returned Purchase
2016-07-18,15 JUL CR INTEREST,-72.97,2016-08-12,Returned Purchase
2016-07-15,13 JUL GRAIN - GO106604 SINGAPORE,45.75,2016-08-12,Delivery
2016-07-18,CGH CLINICS $110.12 001/003,75,2016-08-12,Doctor
2016-07-18,CGH CLINICS $110.12 002/003,75,2016-08-12,Doctor
2016-07-18,CGH CLINICS $110.12 003/003,75,2016-08-12,Doctor
'''
        response = self.lg.handle_request(method='GET', path='/transactions',
                headers={}, body='')
        self.assert_str_as_dataframe_equal(io.StringIO(response['body']),
                                           io.StringIO(expected_payload))

    def test_update_invalid_payload(self):
        response = self.lg.handle_request(method='POST', path='/update',
                headers={'Content-type': 'application/json'}, body='')
        self.assertEqual(response['statusCode'], 400)

        incomplete_update_payload = json.dumps({'description': 'foo'})
        response = self.lg.handle_request(method='POST', path='/update',
                headers={'Content-type': 'application/json'},
                body=incomplete_update_payload)
        self.assertEqual(response['statusCode'], 400)

    def test_request(self):
        password = '123abc'
        payload = self.get_pdf_payload(password)
        response = self.lg.handle_request(method='POST', path='/request',
                headers={'Content-Type': payload.content_type}, body=payload.to_string())

        self.assertEqual(response['statusCode'], 201)
        expected_tags = [{'Key': 'password', 'Value': password},
                         {'Key': 'uuid', 'Value': 'wei'}]
        self.check_and_cleanup_pdf(bucket='moneycat-request-pdfs-dev', expected_tags=expected_tags)

    def test_categories(self):
        response = self.lg.handle_request(method='GET', path='/categories',
                headers={}, body='')

        self.assertEqual(response['statusCode'], 200)
        expected_json = os.path.join(os.path.dirname(__file__), 'data', 'categories.json')
        expected_body = self.read_and_close(expected_json)
        self.assertEqual(response['body'], expected_body)

    def test_categories_etag_cache_header(self):
        response = self.lg.handle_request(method='GET', path='/categories',
                headers={}, body='')

        expected_json = os.path.join(os.path.dirname(__file__), 'data', 'categories.json')
        expected_body = self.read_and_close(expected_json)
        expected_etag = hashlib.md5(expected_body.encode('utf-8')).hexdigest()
        self.assertEqual(response['headers']['ETag'], expected_etag)

        response = self.lg.handle_request(method='GET', path='/categories',
                headers={'If-None-Match': expected_etag}, body='')
        self.assertEqual(response['statusCode'], 304)

    #### helper functions ###
    def get_pdf_payload(self, password=None):
        args = {'file': ('uob.pdf', self.get_pdf_data(), 'application/pdf')}
        if password:
            args['password'] = password
        return MultipartEncoder(args)

    def get_pdf_data(self):
        return self.read_and_close(os.path.join(os.path.dirname(__file__), 'data', 'uob.pdf'), 'rb')

    def check_and_cleanup_pdf(self, bucket='moneycat-pdfs-dev', expected_tags=None):
        # check that pdf is saved on s3
        pdfs = s3.list_objects(Bucket=bucket)['Contents']
        latest_pdf = sorted(pdfs, key=lambda k: k['LastModified'])[-1]
        pdf_obj = s3.get_object(Bucket=bucket, Key=latest_pdf['Key'])
        self.assertEqual(pdf_obj['Body'].read(), self.get_pdf_data())

        if expected_tags:
            self.assertEqual(pdf_obj['TagCount'], len(expected_tags))
            tags = s3.get_object_tagging(Bucket=bucket, Key=latest_pdf['Key'])['TagSet']
            self.assertEqual(tags, expected_tags)

        # only clean up if test passes, when it fails save pdf for debugging
        s3.delete_object(Bucket=bucket, Key=latest_pdf['Key'])

    def read_and_close(self, filename, mode='r'):
        with open(filename, mode) as f:
            return f.read()

    def delete_all_tx_of_test_user(self, get_request_headers={},
            response_body_parse_fn=None):
        response = self.lg.handle_request(method='GET', path='/transactions?txid=1',
                headers=get_request_headers, body='')
        self.assertEqual(response['statusCode'], 200)

        body = response['body']
        if not body or not body.strip():
            print('No data to delete')
            return
        if not response_body_parse_fn:
            response_body_parse_fn = lambda body: pd.read_csv(body, index_col=False)
        tx_df = response_body_parse_fn(io.StringIO(body))
        if tx_df.empty:
            print('No data to delete')
            return

        requests = []
        for index, row in tx_df.iterrows():
            item = {"uuid": {"S": "wei"},
                    "txid": {"S": row['txid']}}
            requests.append({"DeleteRequest": {"Key": item}})
            if len(requests) == 25:
                send_write_request(requests)
                requests = [] # reset requests buffer for next batch
        if len(requests) > 0:
            send_write_request(requests)

    def assert_str_as_dataframe_equal(self, actual_str_io, expected_str_io,
                                      str_parse_fn=None):
        if not str_parse_fn:
            str_parse_fn = lambda body: pd.read_csv(body, index_col=False)
        expected_df = str_parse_fn(expected_str_io) \
                .sort_values(['date', 'description', 'amount']) \
                .reset_index(drop=True)
        if 'foreign_amount' in expected_df:
            expected_df.drop(columns=['foreign_amount'], inplace=True)
        df = str_parse_fn(actual_str_io) \
                .sort_values(['date', 'description', 'amount']) \
                .reset_index(drop=True)
        assert_frame_equal(df, expected_df, check_like=True)


if __name__ == '__main__':
    unittest.main()

