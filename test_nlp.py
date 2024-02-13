'''
    unite tests for class NLPPipeline.
'''
import unittest
from unittest.mock import MagicMock, patch
from nlp_pipeline import NLPPipeline

class TestNLPPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = NLPPipeline()

    def test_detect_language(self):
        text = "This is a test sentence."
        language = self.pipeline.detect_language(text)
        self.assertEqual(language, "en")

    @patch('nlp_pipeline.spacy')
    def test_load_nlp_with_language(self, mock_spacy):
        mock_spacy.load = MagicMock()
        self.pipeline.load_nlp_with_language()
        mock_spacy.load.assert_called_once_with("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

    @patch('nlp_pipeline.spacy')
    def test_get_ner(self, mock_spacy):
        mock_spacy_obj = MagicMock()
        mock_spacy_obj.ents = ['Organization', 'Person']
        mock_spacy.return_value = mock_spacy_obj

        text = "This is a test sentence. Apple and Ali"
        entities = self.pipeline.get_ner(text)
        self.assertEqual([ent.text for ent in entities], ['Apple', 'Ali'])

    @patch('nlp_pipeline.spacy')
    def test_find_dates(self, mock_spacy):
        mock_spacy_obj = MagicMock()
        mock_ent1 = MagicMock()
        mock_ent2 = MagicMock()
        mock_ent1.label_ = "DATE"
        mock_ent2.label_ = "PERSON"
        mock_ent1.text = "2023-02-01"
        mock_ent2.text = "John"
        mock_spacy_obj.ents = [mock_ent1, mock_ent2]
        mock_spacy.return_value = mock_spacy_obj

        text = "John was born on 2023-02-01."
        dates = self.pipeline.find_dates(text)
        self.assertEqual(dates, ["2023-02-01"])

if __name__ == '__main__':
    unittest.main()