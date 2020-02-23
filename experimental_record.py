import wave
import pyaudio
import keyboard
import time
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import time
def print_pressed_keys(e):
    global bool_space
    global bool_esc
    if e.name=='space' and e.event_type=='down':
        bool_space=True    
    if e.name=='esc' and e.event_type=='down':
        bool_esc=True     
    return  


def quick_search(array,value):
    index=len(array)//2
    l=0
    r=len(array)-1
    while True:
        if round(array[index])==value:
            return(index)
        elif round(array[index])>value:
            r=index
            index=r-(r-l)//2
        else:
            l=index
            index=l+(r-l)//2

def frequency_sepectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)

def define_whistle(fft_X,frequency):
    if fft_X.index(max(fft_X[3000:])) in range(frequency-1500,frequency+1500):
        return True

def fft_sig(file):
# wav sample from https://freewavesamples.com/files/Alesis-Sanctuary-QCard-Crickets.wav
#here_path = os.path.dirname(os.path.realpath(__file__))
#wav_file_name = 'C:\Work\speech_recognition\Week_1\samples\sample_2.wav'
#here_path = os.path.dirname(os.path.realpath(__file__))
#wav_file_name = 'sample_3.wav'
#wave_file_path = os.path.join(here_path, wav_file_name)
    sr, signal = wavfile.read(file)

    y = signal[:,0] # use the first channel (or take their average, alternatively)   
    frq, X = frequency_sepectrum(y, sr)
    
    
    #a=quick_search(frq,3300)
    #if define_whistle(list(X),a):
    #    print("It's a whistle")
    #else:
    #    print("It isn't a whistle")
    return [X,frq]

def record(p,file):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = file

    p_1 = pyaudio.PyAudio()

    stream_1 = p_1.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []
    time_1=int(RATE / CHUNK * RECORD_SECONDS)
    for i in range(0, time_1):
        data = stream_1.read(CHUNK)
        frames.append(data)            #print("* done recording")
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        if i in range(100, time_1,100):
            print(1)
            a=fft_sig(WAVE_OUTPUT_FILENAME)
            if define_whistle(list(a[0]),quick_search(a[1],3300)):
                frames=[]
                p_2 = pyaudio.PyAudio()
                stream_2 = p_2.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)
                print("* recording")
                time_2 = int(RATE / CHUNK * 3)
                for i in range(0, time_2):
                    data = stream_2.read(CHUNK)
                    frames.append(data)
                    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                stream_2.stop_stream()
                stream_2.close()
                p_2.terminate()
                print('*done recording')
    stream_1.stop_stream()
    stream_1.close()
    p_1.terminate()
    print('*done recording')
    return frames[0]
    
#p1 = pyaudio.PyAudio()
#p2 = pyaudio.PyAudio()
#record(p1,'C:\Work\Experiments\sample.wav')
#record(p2,'C:\Work\Experiments\sample2.wav')
def found_n():
    n=1
    while True:
        try:
            open('C:\Work\speech_recognition\sit\sample_{}.wav'.format(n),'r')
        except FileNotFoundError:
            break
        n+=1
    return n

def main():
    global bool_space
    global bool_esc
    n=found_n()
    bool_space=False
    bool_esc=False
    keyboard.hook(print_pressed_keys)
    
    while True:
        if bool_esc:
            break
        if bool_space: 
            p = pyaudio.PyAudio()
            #sample = 'C:\Work\speech_recognition\Week_2\samples_with_\sample_{}.wav'.format(n)
            sample = 'C:\Work\speech_recognition\sit\sample_{}.wav'.format(n)
            #keyboard.add_hotkey('space', lambda: record(p,sample)) 
            record(p,sample)  
            n+=1
            bool_space=False
            
if __name__ == "__main__":
    main() 
