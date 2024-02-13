from typing import Any
import gradio as gr
from chatbot import ChatBot
import logging
if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    chatbot =ChatBot()
    gr.ChatInterface(chatbot.chat_gradio).launch()