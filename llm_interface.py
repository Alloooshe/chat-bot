from litellm import completion
import json 
import logging

class LLMInference:
    """This is a base class for inference from local LLM.
    """
    def __init__(self, base_url="http://localhost:11434/",api_key="ollama") -> None:
        self.base_url = base_url
        self.api_key= api_key
    

    def completion(self,**kwds):
        """ 
            This function is a wrapper to calls text completion method.  

        Returns:
            ModelResponse: A response object containing the generated completion and associated metadata.
        """
        kwds["base_url"]=self.base_url
        kwds["api_key"]=self.api_key
        return completion(
           **kwds
        )
    
class ChatAgent(LLMInference):
    """
        This class implements LLMInference class and adds logic to produce a chat agent.   

    Args:
        LLMInference (LLMInference): base class for LLM inference.
    """
    def __init__(self,conf={
                "model":"ollama/llama2",
            }, base_url="http://localhost:11434", api_key="ollama") -> None:
        super().__init__(base_url, api_key)
        self.conf =conf
       
    def __call__(self, **kwds):
        return super().completion(**kwds)
    
    def convert_gio_2_to_openai(self,history):
        """
            Helper function to convert chat hisotry from gradio format to openai format.

        Args:
            history (List): list of earlier messages in gradio format.

        Returns:
            List: list of earlier messages in openai format.
        """
        logging.info("converting history to openai format")
        history_openai_format = []
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human })
            history_openai_format.append({"role": "assistant", "content":assistant})
        return history_openai_format
    
    def chat_stream(self,message,history,additional_context):
        """
            This function implements logic to handle chat with streamed output using LLM.        

        Args:
            message (string): user's input
            history (List): list of earlier messages in gradio format.
            additional_context (string): additional context to add to history chat under role user.

        Yields:
            string: the LLM's chunked response. 
        """
        history_openai_format = self.convert_gio_2_to_openai(history)
        if len(additional_context)>0:
            logging.info("adding additional context")
            message = f"{message} additonal info {additional_context}"
        history_openai_format.append({"role": "user", "content": message})
        self.conf["messages"]= history_openai_format
        self.conf["stream"]=True
        response = self.__call__(**self.conf)
        res=""
        logging.info("start streaming response")
        for chunk in response:
            if chunk.choices[0].delta.content is not None and len(chunk.choices[0].delta.content) >0 :
                res += chunk.choices[0].delta.content
                yield res
                
    def chat(self,messages,additional_context):
        """
            This function implements logic to handle chat without streaming  using LLM.      

        Args:
            messages (List): List of messages in openai format.
            additional_context (string): additional context to add to history chat under role user.

        Returns:
            ModelResponse: A response object containing the generated completion and associated metadata.
        """
        if len(additional_context)>0:
            logging.info("adding additional context")
            messages[0]['content'] = f"{messages[0]['content']} \n additonal info: {additional_context}"
        self.conf["messages"]= messages
        self.conf["stream"]=False
        return self.__call__(**self.conf)

class IntentClassifier(LLMInference):
    """
        This class implements LLMInference class and adds logic to produce a intent classifier using function calls.   

    Args:
        LLMInference (LLMInference): base class for LLM inference.
    """
    def __init__(self,intents=["search_documents","chat","request_help","close_order","request_human_operator","other"],
                conf={
                    "model":"ollama/llama2",
                    "max_tokens":300,
                    "top_p":0.5,
                    "temperature":0.5,
                    "messages":[]
                }, 
                 base_url="http://localhost:11434",
                 api_key="ollama") :
        
        super().__init__(base_url, api_key)
        self.conf=conf
        self.intents =intents
        self.tool_choice= {"type:":"function", "function": {"name": "parse_request"}}
        self.tools = [
                {
                    "type" : "function",
                    "function" : {
                                    "name": "parse_request",
                                    "description": "parse user's request and print a normal response.",
                                    "parameters": {
                                    "type": "object",
                                    "properties": {
                                    
                                        "intent": {
                                        "type": "string",
                                        "enum": self.intents,
                                        "description": "user's intent with the request."
                                        },
                                    },
                                    "required": ["intent"],
                                    }
                                }
                }
            ]
        self.system_message=  {
            "role": "system",
            "content": "you will recieve a request from user's chat, you need to parse the request using parse_request function."
            }
    
    def __call__(self, **kwds):
        return super().completion(**kwds)
    @staticmethod
    def find_key_by_value(dictionary, target_value):
        """
            helper function to parse unstable output of function calls

        Args:
            dictionary (dict): nested dectionary of values.
            target_value (string): the target key you want to search for.

        Returns:
            string: value of the target key, or None if nothing was found.
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                result = IntentClassifier.find_key_by_value(value, target_value)
                if result is not None:
                    return result
            elif key == target_value: 
                return value
        return None

    def detect_intent(self,message):
        """
            This fuction determine the user's intent in text, the list of intents is provided on class initalization.

        Args:
            message (string): user's input.

        Returns:
            string: user's intent enum [self.intents]
        """
        self.conf["tools"]= self.tools
        self.conf["tool_choice"]= self.tool_choice
        self.conf["messages"].append(self.system_message)
        self.conf["messages"].append({"role": "user", "content": message})
        response = self.__call__(**self.conf)
        ret = None 
        try : 
            logging.info("attemting to parse function call response")
            response_dict =json.loads(response['choices'][0]['message']['content'])['response']
            ret =  IntentClassifier.find_key_by_value(response_dict, 'intent') 
        except Exception as e : 
            logging.error("could not parse intent classifier response.")
        ret = ret if ret is not None else "NA"
        return ret 
    