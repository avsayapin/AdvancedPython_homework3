import unittest
from app.api import app


class TestAPP(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def TestClasses(self):
        response = self.client.get("/classes")
        self.assertIn(b'{"classes":["Linear regression","Gradient Boosting regression","Logistic regression","Gradient Boosting classifier"]}\n', response.data)
        self.assertEqual(response.status_code, 200)
