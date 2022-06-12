import numpy as np
import gradio as gr
from main import *

timestamp_size = 2048
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


def transcribe(mic, speech):

    if mic:
        sr, y = mic
        rate, data = reformat_freq(sr, y)
        print("Mic ", (rate, data))

    if speech:
        print("Recorded ", speech)
        rate, data = speech

    filtered_sig = voice_denoise(data, rate, model, 0, timestamp_size, numcep=20, plot=True)
    output = (rate, np.asarray(filtered_sig * 32768, dtype=np.int16))
    return output



gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="numpy"), 
        gr.inputs.Audio(),
    ], 
    outputs= [
        "audio",
    ], 
    ).launch()