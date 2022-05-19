import random
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment


def time_stretch(signal, time_stretch_rate):
    """Time stretching implemented with librosa:
    https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
    """
    return librosa.effects.time_stretch(signal, time_stretch_rate)


def pitch_scale(signal, sr, num_semitones):
    """Pitch scaling implemented with librosa:
    https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
    """
    return librosa.effects.pitch_shift(signal, sr, num_semitones)

def invert_polarity(signal):
    return signal * -1

def augment():
    for i in range(62):
        signal, sr = librosa.load('MS-SNSD/CleanSpeech_training/clnsp'+str(i+1)+'.wav')
        #augmented_signal = time_stretch(signal,0.5)
        augmented_signal = pitch_scale(signal,sr,4)
        sf.write("MS-SNSD/CleanSpeech_training/clnsp"+str(i+63)+".wav", augmented_signal, sr)
    
    j=0
    i=0
    while(j<62):
        signal, sr = librosa.load('MS-SNSD/CleanSpeech_training/clnsp'+str(j+1)+'.wav')
        #augmented_signal = time_stretch(signal,0.5)
        augmented_signal = time_stretch(signal,0.5)
    
    
        sf.write("MS-SNSD/temp_time_stretching/temp"+str(i)+".wav", augmented_signal, sr)
        t1 = 10000 #Works in milliseconds
        t2 = 20000 #Works in milliseconds
    
        newAudio = AudioSegment.from_wav("MS-SNSD/temp_time_stretching/temp"+str(i)+".wav")
        FirstHalf = newAudio[0:t1]
        SecondHalf = newAudio[t1:t2]
        FirstHalf.export("MS-SNSD/CleanSpeech_training/clnsp"+str(i+125)+".wav", format="wav")
        i+=1
        SecondHalf.export("MS-SNSD/CleanSpeech_training/clnsp"+str(i+125)+".wav", format="wav")
        i+=1
        j+=1    