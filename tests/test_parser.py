import unittest
from sentinel import LogParser, FeatureExtractor


class TestParserAndFeaturizer(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()
        self.fe = FeatureExtractor()
        self.sample_line = '127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 1024'

    def test_parse_line(self):
        parsed = self.parser.parse_line(self.sample_line)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['ip'], '127.0.0.1')
        self.assertEqual(parsed['status_code'], 200)
        self.assertEqual(parsed['response_size'], 1024)

    def test_featurize(self):
        parsed = self.parser.parse_line(self.sample_line)
        features = self.fe.featurize(parsed)
        self.assertEqual(len(features), 4)
        self.assertEqual(features[0], 200)  # status code
        self.assertEqual(features[2], 1024)  # response size



if __name__ == '__main__':
    unittest.main()
