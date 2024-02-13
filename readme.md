## General overview 
we implement a chatbot capable of more than simply chat. 

## setup 
* download and install ollama [docker image](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image)

* choose a model and pull it, they current code assums the LLM runs on  http://localhost:11434/ feel free to modify that. 

* create python enviroment
* installl requirments.txt

## Architecture
* LLM inference base class we have 
* NLP pipeline (with unit tests)
* chatbot with logic 
* server FLASK + usage example : gradio chat UI client 

## functionality 
### NLP pipeline
* language detection 
* Named Entity Recognition using spacy: the code will try to load the best spacy model for the detected language after each request.
* dates extraction : we reuse the last step and extract all detected dates.
* Intent classification : this is very domain spesfic task: ideally you will have an embedding model and classification model. **for embedding it is possible to use off the shelf model or even LLM** for classification it is better to train a different model  like **BERT** to classify the chat or the input into intent classes. 
to simulate this behaviuor I reused the deployed LLM with **function calling** aka I used LLM as a zero-shot intent classifier. This is not optimal but it provides a quick solution. 
* LLM choice : as mentioed before I wanted to use function calling, so it is better to use a model trained on function calling like [nexusraven](https://ollama.com/library/nexusraven) however this model is heavy 13B parameters and I prefered the much lighter [llama2](https://ollama.com/library/llama2) which is optimized for chats. 
* LLM and function calling : using a LLM which was not trained for function calling results in suboptimal performance and incosistant results. but again it is a trade-off 

### Chat with LLM
I implemented chat completion function and chat _streaming_ using litellm library as an interface for Ollama.

### NLU 
The chatbot however uses the NLP pipeline to prompt the answers, I included an example where if the detected user intent is search_document then it will add information about entites and dates to the prompt which will help the LLM to produce more accurate response. this is not optimal NLU but it simulates the process.

## Usage
* run the Flask server using server.py don't forget to include SERVER_TOKEN in your enviroment for authentication. the endpoint chat accepts openai style input and returns a dictinary of response and meta data (NLP pipeline info)
 
* run chatbot UI using gradio : simply run client_local.py, it should open a chat interface in the browser.

