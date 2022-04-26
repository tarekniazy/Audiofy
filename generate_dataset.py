import scipy.io.wavfile as wav
import feature_extraction as ft
import os
import numpy as np
import re


def gen_dataset(path,vad_active_delay=0.07,vad_threshold=1e-1,winstep=0.032/2):

    mfcc_data = []
    total_energy = []
    band_energy = []
    vad = []
    filename_label = []

    files = os.listdir(path)
    for f in files:
        filename = f
        if ('wav' not in filename):
            continue
        (rate, sig) = wav.read(path+'/'+f)
        # convert file to [-1, 1)
        sig = sig/32768  
    
        mfcc_feat,band_eng, total_eng=ft.mfcc_trial(sig,rate)
        
        sig = sig * np.random.uniform(0.8, 1)

        v = (total_eng > vad_threshold).astype(int)
        vad_delay = int(vad_active_delay*(rate*winstep))
        conv_win = np.concatenate([np.zeros(vad_delay), np.ones(vad_delay)]) # delay the VAD for a vad_active_delay second
        v = np.convolve(v, conv_win, mode='same')
        v = (v > 0).astype(int)

        total_energy.append(total_eng)
        band_energy.append(band_eng)
        vad.append(v)
        mfcc_data.append(mfcc_feat.astype('float32'))

        filename_label.append(filename)


    return mfcc_data, filename_label,band_energy, total_energy,vad 


if __name__ == "__main__":

    vad_energy_threashold = 0.1
    noisy_speech_dir = 'MS-SNSD/NoisySpeech_training'
    clean_speech_dir = 'MS-SNSD/CleanSpeech_training'
    noise_dir = 'MS-SNSD/Noise_training'


    clean_speech_mfcc, clean_file_label, clnsp_band_energy,total_energy, vad  = gen_dataset(clean_speech_dir,vad_threshold=vad_energy_threashold)

    # add noise to clean speech, then generate the noise MFCC
    print('generating noisy speech MFCC...')
    noisy_speech_mfcc, noisy_file_label,noisy_band_energy, _, _   = gen_dataset(noisy_speech_dir,vad_threshold=vad_energy_threashold)

    # noise_only_mfcc, noise_only_label, _, _ , noise_band_energy = gen_dataset(noise_dir,vad_threshold=vad_energy_threashold)

    clnsp_mfcc = []
    noisy_mfcc = []
    noise_mfcc = []
    voice_active = []
    gains_array = []

    print('Processing training data')

    print(len(clnsp_band_energy))

    print(len(noisy_band_energy))

    for idx_nosiy, label in enumerate(noisy_file_label):
        # get file encode from file name e.g. "noisy614_SNRdb_30.0_clnsp614.wav"
        nums = re.findall(r'\d+', label)
        file_code = nums[0]
        db_code = nums[1]


        idx_clnsp = clean_file_label.index('clnsp'+str(file_code)+'.wav')

        


        gains = np.sqrt(clnsp_band_energy[idx_clnsp]/ noisy_band_energy[idx_nosiy])
        gains = np.clip(gains, 0, 1)

        voice_active.append(vad[idx_clnsp])
        clnsp_mfcc.append(clean_speech_mfcc[idx_clnsp])
        # noise_mfcc.append(noise_only_mfcc[idx_nosiy])
        noisy_mfcc.append(noisy_speech_mfcc[idx_nosiy])
        gains_array.append(gains)

    # save the dataset.

    # print(np.array(clnsp_mfcc).shape)
    # print(np.array(noisy_mfcc).shape)
    # print(np.array(noise_mfcc).shape)
    # print(np.array(voice_active).shape)
    # print(np.array(gains_array).shape)

    np.savez("dataset2.npz", clnsp_mfcc=clnsp_mfcc, noisy_mfcc=noisy_mfcc,  vad=voice_active, gains=gains_array)
    print("Dataset generation has been saved to:", "dataset.npz")