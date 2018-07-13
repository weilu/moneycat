import unittest
import os
import base64
from chalice.config import Config
from chalice.local import LocalGateway
from app import app, s3
from requests_toolbelt import MultipartEncoder

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
        self.assertEqual(pdf_obj['TagCount'], 2)

        if expected_tags:
            tags = s3.get_object_tagging(Bucket=bucket, Key=latest_pdf['Key'])['TagSet']
            self.assertEqual(tags, expected_tags)

        # only clean up if test passes, when it fails save pdf for debugging
        s3.delete_object(Bucket=bucket, Key=latest_pdf['Key'])

    def read_and_close(self, filename, mode='r'):
        with open(filename, mode) as f:
            return f.read()


if __name__ == '__main__':
    unittest.main()

