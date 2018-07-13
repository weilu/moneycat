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
        self.assertEqual(response['body'], open(expected_csv).read())
        self.check_and_cleanup_pdf()

    def test_upload_json(self):
        payload = self.get_pdf_payload()
        response = self.lg.handle_request(method='POST', path='/upload',
                headers={'Content-Type': payload.content_type,
                         'Accept': 'application/json'},
                body=payload.to_string())

        self.assertEqual(response['statusCode'], 200)
        expected_json = os.path.join(os.path.dirname(__file__), 'data', 'uob.json')
        self.assertEqual(response['body'], open(expected_json).read().strip())
        self.check_and_cleanup_pdf()

    def test_upload_with_password(self):
        payload = self.get_pdf_payload('123abc')
        response = self.lg.handle_request(method='POST', path='/upload',
                headers={'Content-Type': payload.content_type}, body=payload.to_string())

        self.assertEqual(response['statusCode'], 200)
        expected_csv = os.path.join(os.path.dirname(__file__), 'data', 'uob.csv')
        self.assertEqual(response['body'], open(expected_csv).read())
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


    def get_pdf_payload(self, password=None):
        args = {'file': ('uob.pdf', self.get_pdf_data(), 'application/pdf')}
        if password:
            args['password'] = password
        return MultipartEncoder(args)

    def get_pdf_data(self):
        return open(os.path.join(os.path.dirname(__file__), 'data', 'uob.pdf'), 'rb')

    def check_and_cleanup_pdf(self):
        # check that pdf is saved on s3
        bucket = 'moneycat-pdfs-dev'
        pdfs = s3.list_objects(Bucket=bucket)['Contents']
        latest_pdf = sorted(pdfs, key=lambda k: k['LastModified'])[-1]
        model_obj = s3.get_object(Bucket=bucket, Key=latest_pdf['Key'])
        self.assertEqual(model_obj['Body'].read(), self.get_pdf_data().read())

        # only clean up if test passes, when it fails save pdf for debugging
        s3.delete_object(Bucket=bucket, Key=latest_pdf['Key'])

if __name__ == '__main__':
    unittest.main()

