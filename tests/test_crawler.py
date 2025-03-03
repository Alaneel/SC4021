# tests/test_crawler.py
import unittest
from crawler.data_cleaner import clean_text


class TestDataCleaner(unittest.TestCase):
    def test_clean_text(self):
        # Test with URLs
        text = "Check out this link: https://example.com"
        cleaned = clean_text(text)
        self.assertNotIn("https://", cleaned)

        # Test with empty text
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")

        # Test with special characters
        text = "Electric cars are amazing!!!! #EV #sustainability"
        cleaned = clean_text(text)
        self.assertIn("Electric cars are amazing", cleaned)
        self.assertNotIn("#", cleaned)


if __name__ == '__main__':
    unittest.main()