import unittest
import os
import base64
from chalice.config import Config
from chalice.local import LocalGateway
from app import app
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


    def get_pdf_payload(self):
        pdf_data = open(os.path.join(os.path.dirname(__file__), 'data', 'uob.pdf'), 'rb')
        return MultipartEncoder({'file': ('uob.pdf', pdf_data, 'application/pdf')})


if __name__ == '__main__':
    unittest.main()

