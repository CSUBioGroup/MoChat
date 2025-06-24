import torch
import gradio as gr
from fastapi import FastAPI
from MoChat.conversation import conv_templates, Conversation
from MoChat.demo import Chat
from MoChat.constants import *
import os
import tempfile
import shutil
import numpy as np


app = FastAPI()
model_path = "" 

def skeleton_generate(skeleton, textbox_in, first_run, state, state_):
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    skeleton = skeleton if skeleton else "none"

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        images_tensor = []

    first_run = False if len(state.messages) > 0 else True

    text_en_in = textbox_in.replace("picture", "skeleton")

    skeleton = torch.from_numpy(skeleton).float()
    C, T, V, M = skeleton.shape
    skeleton = skeleton.contiguous().view(C,T,V*M)
    images_tensor.append(skeleton)

    text_en_out, state_ = handler.generate(images_tensor, text_en_in, first_run=first_run, state=state_)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out

    if flag:
        state.append_message(state.roles[0], textbox_in)
    state.append_message(state.roles[1], textbox_out)

    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True))

def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True), \
        gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        True, state, state_, state.to_gradio_chatbot(), [])


conv_mode = "simple"
handler = Chat(model_path, conv_mode=conv_mode)
if not os.path.exists("temp"):
    os.makedirs("temp")

with gr.Blocks(gr.themes.Soft()) as demo:
    demo.title = 'Demo'
    state = gr.State()
    state_ = gr.State()
    first_run = gr.State()
    images_tensor = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            skeleton_array = np.load(skeleton_path)

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(label="Chat-UniVi", bubble_full_width=True).style(height=1200)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox = gr.Textbox(label="Input Text")
                with gr.Column(scale=1, min_width=60, label="Input Text"):
                    submit_btn = gr.Button(value="Submit", visible=True)
            with gr.Row(visible=True) as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

    submit_btn.click(skeleton_generate, [skeleton_array, textbox, first_run, state, state_, images_tensor], [state, state_, chatbot, first_run, textbox, images_tensor, image1, image2, video])

    regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
        skeleton_generate, [skeleton_array, textbox, first_run, state, state_, images_tensor], [state, state_, chatbot, first_run, textbox, images_tensor, image1, image2, video])

    clear_btn.click(clear_history, [state, state_],
                    [skeleton_array, textbox, first_run, state, state_, chatbot, images_tensor])

app = gr.mount_gradio_app(app, demo, path="/")
