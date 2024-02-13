from nlp_pipeline import NLPPipeline
from llm_interface import ChatAgent, IntentClassifier
import logging 
class ChatBot:
    def __init__(self) -> None:
        self.intents = ["search_documents","chat","request_help","close_order","request_human_operator","other"]
        self.nlp_processor = NLPPipeline()
        self.intent_classifier= IntentClassifier(intents=self.intents)
        self.chat_agent = ChatAgent()
    
    def add_scenario_context(self, meta_data):
        """
            Helper function that generates useful context for a LLM, it uses information extracted from the NLP pipeline and the intent classifier. 

        Args:
            meta_data (dict): meta data information which can be used to generate additional context.

        Returns:
            string: prompt for LLM.
        """
        logging.info("adding scenario context")
        additional_context= f"chat language {meta_data['language']}. "
        if meta_data['intent'] == "search_documents": 
            additional_context+= "\nintent : search for document. " 
            additional_context+= f"infomation about the document: entities [{', '.join([ent['text'] for ent in meta_data['ner_ents']])}]. dates : [{', '.join(meta_data['dates'])}].\n"
        return additional_context
    
    def chatbot_logic(self,message):
        """
            This function is a Brain function that handles a chat with a user. it is where you can implement chatbot logic.

        Args:
            message (string): user's input

        Returns:
            Tuple(dict,string): meta data dictionary and additioanl context string. 
        """
        logging.info("starting chatbot logic")
        meta_data = {}
        meta_data["language"]= self.nlp_processor.detect_language(message)
        meta_data ["ner_ents"] = self.nlp_processor.ner_to_dict(self.nlp_processor.get_ner(message))
        meta_data ["dates"]= self.nlp_processor.find_dates(message)
        if meta_data["language"] !="en":
            meta_data ["intent"] ="UNKOWN"    
        else :
            meta_data ["intent"] = self.intent_classifier.detect_intent(message)
        
        logging.info(f"detected intent is {meta_data['intent']}")
        additional_context = self.add_scenario_context(meta_data)
        return meta_data, additional_context
    
    def chat_gradio(self,message,history):
        """This function implements a chat function for testing with gradio.

        Args:
            message (string): user's input
            history (List): chat history in gradio format

        Yields:
            string: the LLM's chunked response. 
        """
        _, additional_context = self.chatbot_logic(message)
        logging.info(f"additional context {additional_context}")
        for chunk in  self.chat_agent.chat_stream(message,history,additional_context):
            yield chunk
    
    def chat(self,messages):
        """This function implements a chat example with meta data.

        Args:
            messages (string): List of messages in openai format.

        Returns:
           dict: dictionary with two keys : output (LLM's response), meta_data (NLP pipeline output).
        """
        meta_data, additional_context = self.chatbot_logic(messages["messages"][-1]["content"])
        logging.info(f"additional context {additional_context}")
        logging.info(f"meta data {meta_data}")
        output = self.chat_agent.chat(messages["messages"],additional_context)
        return {"output":output.choices[0].message.content,"meta_data":meta_data}

