from flask import Flask, request, abort, make_response
from chatbot import ChatBot
import logging
import os 
import json

app = Flask(__name__)
chat_bot = ChatBot()

SERVER_TOKEN = os.getenv('SERVER_TOKEN')

@app.before_request
def check_special_token():
    logging.info("attempting authentication")
    token = request.headers.get('Authorization')
    if token != f"token {SERVER_TOKEN}" and token != f"Token {SERVER_TOKEN}":
        logging.error("authentication failed")
        abort(401, "Unauthorized")

@app.route('/chat/', methods=['POST'])
def chat(): 
    messages = json.loads(request.form.get('messages'))
    res = chat_bot.chat(messages)
    return make_response(res, 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1194,debug=True)