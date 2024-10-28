import os
import time

from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

from llm_pipeline import get_pipeline, stop_word_list
import gradio as gr
import argparse
import torch

import requests
from PIL import Image
from torchvision import transforms

# load Image model
response = requests.get("model_path")
labels = response.text.split("\n")
model = torch.hub.load('model_path', 'model_version', pretrained=True).eval()

# Image model predict
def predict(inp,history):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
        print(confidences)

    output_text = 'Looking for the food.'
    return "", history + [[None,output_text]], confidences

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        default='meta-llama/Llama-2-7b-chat-hf',
                        help="Directory containing trained actor model")
    parser.add_argument("--peft",
                        type=int,
                        default=0,
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    return args



def main(args):
    # load Chatbot model
    model_id = args.path
    llm = HuggingFacePipeline(pipeline=get_pipeline(model_id, args.peft))
    # template for Chatbot
    template = """Assistant is a large language model.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    Note: Do NOT be tempted to generate human response.

    Current conversation:
    {history}
    Human: {input}
    AI:"""


    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    # load lang_chain for memory management
    llm_chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )

    custom_css = """
        #chat-bot {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
            overflow: auto;
        }
        #ner-extract {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
            overflow: auto;
        }
    """
    # start and load chat history
    def user(user_message, history):
        return "", history + [[user_message, None]]
    # manage Chatbot history
    def bot(history):
        bot_message = llm_chain.run(history[-1][0])
        for stop_word in stop_word_list:
            if bot_message.rfind(stop_word) != -1:
                bot_message = bot_message[: -len(stop_word)]
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.005)
            yield history
    # clear Chatbot memory
    def clear_mem():
        llm_chain.memory.clear()
    # Chatbot load and explain about the dish
    def bot_food(history, image_outputs):
        input_text = 'here is ' + image_outputs['label'] +'.\n tell me about it.'
        bot_message = llm_chain.run(input_text)
        for stop_word in stop_word_list:
            if bot_message.rfind(stop_word) != -1:
                bot_message = bot_message[: -len(stop_word)]
        history[-1][1] = "This is " + image_outputs['label'] + '.\n\n'

        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.005)
            yield history
    # UI block design
    with gr.Blocks(css=custom_css) as demo:
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(elem_id="chat-bot", show_copy_button=True,render_markdown=False)
            with gr.Column():
                image_inputs = gr.Image(type="pil")
                image_outputs = gr.Label(num_top_classes=3)
                image_process = gr.Button(value="Predict Food")
        msg = gr.Textbox()
        with gr.Row():
            with gr.Column():
                clear = gr.Button("clear")

        # make action button live
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        image_process.click(fn=predict, inputs=[image_inputs,chatbot], outputs=[msg, chatbot,image_outputs]).then(bot_food, [chatbot,image_outputs], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False).then(clear_mem)
    demo.queue()
    demo.launch(server_port=7860, server_name="0.0.0.0")
    demo.launch()

if __name__ == "__main__":

    args = parse_args()
    main(args)