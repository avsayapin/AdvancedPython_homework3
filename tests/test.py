from app.api import app
import unittest


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client(self)

    def testClasses(self):
        response = self.client.get('/classes')
        self.assertIn(b'{"classes":["Linear regression","Gradient Boosting regression","Logistic regression",'
                      b'"Gradient Boosting classifier"]}\n',
                      response.data)
        self.assertEqual(response.status_code, 200)

    def testModels(self):
        response = self.client.get('/models')
        self.assertIn(b'{"models":[]}\n', response.data)
        self.assertEqual(response.status_code, 200)
