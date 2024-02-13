import logging
import spacy 
from lingua import LanguageDetectorBuilder

class NLPPipeline:
    """
        This class perfroms several NLP tasks including : NER, Date extraction, and language detection.  
    """
    def __init__(self) -> None:
        self.language_detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
        self.language="en"
        if not "en_core_web_sm" in spacy.info()['pipelines'].keys():
            spacy.cli.download("en_core_web_sm")
        self.nlp_spacy = spacy.load("en_core_web_sm",disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    
    def detect_language(self,text):
        """this function detects the language of the input.

        Args:
            text (string): input string

        Returns:
            string: the main language detected in the text is iso code. for example : en,ar,fr...etc. 
        """
        logging.info("detecing language")
        self.language= self.language_detector.detect_language_of(text).iso_code_639_1.name.lower()
        logging.info(f"detected language is : {self.language}")
        self.load_nlp_with_language()
        return self.language
    
    def load_nlp_with_language(self):
        """
            loads spacy model for pipeline language. if the language is not supported it loads the multilanuage model xx_ent_wiki_sm.
        """
        logging.info("loading spacy language model")
        try:
            logging.info("trying to find spacy model for the detected language.")
            suffix="_core_web_sm"
            if self.language !="en":
                suffix ="_core_news_sm"                 
            new_model_name = f"{self.language}{suffix}"
            if self.nlp_spacy.meta["lang"] == self.language :
                logging.debug("reusing the same model")
                return 
            if spacy.util.is_package(new_model_name):
                logging.debug(f"attempting to download spacy model:{new_model_name} ")
            else : 
                if self.nlp_spacy.meta["lang"] == "xx" :
                    return 
                logging.info(f"could not find model:{new_model_name} reverting to multilanguage model ")
                new_model_name =  "xx_ent_wiki_sm"
            if not new_model_name in spacy.info()['pipelines'].keys():
                spacy.cli.download(new_model_name)
            self.nlp_spacy = spacy.load(new_model_name,disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            
        except Exception as e:
            logging.error(e) 
            logging.debug(f"could not find nlp model for language {self.language}")
        
    def get_ner(self,text):
        """
            This function extracts named entities from a text. 

        Args:
            text (string): input string.

        Returns:
            Span: spacy objects that includes the detected NE.
        """
        logging.info("extracting entites from text.")
        doc = self.nlp_spacy(text )
        return doc.ents
    
    def ner_to_dict(self,ner):
        return [{"text": ent.text,"label":ent.label_} for ent in ner]
    
    def find_dates(self,text):
        """
            This function extracts dates from a text by first performing NER and then choosing entites with label DATE.

        Args:
            text (string): input string

        Returns:
            List: list of detected dates.
        """
        logging.info("extracting dates from entites.")
        entities = self.get_ner(text)
        ret = []
        for ent in entities:
            if ent.label_ =="DATE":
                ret.append(ent.text)
        return ret

if __name__=="__main__": 
    #local test, for debuging only.
    nlppieline =NLPPipeline()
    sentence = "أهلاً! كيف حالك اليوم."
    nlppieline.detect_language(sentence)
    nlppieline.detect_language("Hi! I need information about Alpha. Inc. which was registered in march 2024 by Ali Mohammad.")
    nlppieline.detect_language(sentence)
    nlppieline.get_ner("Hi! I need information about Alpha. Inc. which was registered in march 2024 by Ali Mohammad.")
    nlppieline.find_dates("Hi! how are you today")