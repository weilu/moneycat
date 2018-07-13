import unittest
import os
import io
import base64
import pandas as pd
from chalice.config import Config
from chalice.local import LocalGateway
from app import app, s3, send_write_request, dynamodb
from requests_toolbelt import MultipartEncoder
from pandas.util.testing import assert_frame_equal

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
        # first delete all transactions of the test user's
        response = self.lg.handle_request(method='GET', path='/transactions?txid=1',
                headers=get_request_headers, body='')
        self.assertEqual(response['statusCode'], 200)
        df = response_body_parse_fn(io.StringIO(response['body']))
        self.delete_all_tx_of_test_user(df)

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
        expected_df = response_body_parse_fn(filename) \
                .drop(columns=['foreign_amount']) \
                .sort_values(['date', 'description', 'amount']) \
                .reset_index(drop=True)
        df = response_body_parse_fn(io.StringIO(response['body'])) \
                .sort_values(['date', 'description', 'amount']) \
                .reset_index(drop=True)
        assert_frame_equal(df, expected_df, check_like=True)

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

    def test_request(self):
        password = '123abc'
        payload = self.get_pdf_payload(password)
        response = self.lg.handle_request(method='POST', path='/request',
                headers={'Content-Type': payload.content_type}, body=payload.to_string())

        self.assertEqual(response['statusCode'], 201)
        expected_tags = [{'Key': 'password', 'Value': password},
                         {'Key': 'uuid', 'Value': 'wei'}]
        self.check_and_cleanup_pdf(bucket='moneycat-request-pdfs-dev', expected_tags=expected_tags)


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

    def delete_all_tx_of_test_user(self, tx_df):
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



if __name__ == '__main__':
    unittest.main()

