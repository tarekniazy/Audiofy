import numpy as np
import gradio as gr
from main import *

timestamp_size = 128
model = load_model("model.h5", custom_objects={'mycost': mycost, 'msse':msse, 'my_crossentropy':my_crossentropy, 'my_accuracy':my_accuracy})

def reformat_freq(sr, y):
    if sr not in (
        48000,
        16000,
    ):
        raise ValueError("Unsupported rate", sr)
    if sr == 48000:
        y = (
            ((y / max(np.max(y), 1)) * 32767)
            .reshape((-1, 3))
            .mean(axis=1)
            .astype("int16")
        )
        sr = 16000
    return sr, y


def filter(option, speech, mic):

    if option == "Microphone":
        print("HELLO")
        sr, y = mic
        rate, data = reformat_freq(sr, y)
        print("Mic ", (rate, data))

    if option == "Upload Audio":
        print("HI")
        print("Recorded ", speech)
        rate, data = speech

    filtered_sig = voice_denoise(data, rate, model, timestamp_size, numcep=20, plot=True)
    output = (rate, np.asarray(filtered_sig * 32768, dtype=np.int16))
    return output



gr.Interface(
    fn=filter, 
    inputs=[
        gr.Dropdown(["Upload Audio", "Microphone"]),
        gr.inputs.Audio(),
        gr.inputs.Audio(source="microphone", type="numpy"), 
    ], 
    outputs= [
        "audio"
    ],
    ).launch(share=True)