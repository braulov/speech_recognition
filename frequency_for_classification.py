from scipy import fft, arange
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import time
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
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)

def fft_sig(file):
# wav sample from https://freewavesamples.com/files/Alesis-Sanctuary-QCard-Crickets.wav
#here_path = os.path.dirname(os.path.realpath(__file__))
#wav_file_name = 'C:\Work\speech_recognition\Week_1\samples\sample_2.wav'
#here_path = os.path.dirname(os.path.realpath(__file__))
#wav_file_name = 'sample_3.wav'
#wave_file_path = os.path.join(here_path, wav_file_name)
    sr, signal = wavfile.read(file)

    y = signal[:,0] # use the first channel (or take their average, alternatively)
    t = np.arange(len(y)) / float(sr)
    frq, X = frequency_sepectrum(y, sr)
    
    lower_bound=quick_search(frq,85)
    upper_bound=quick_search(frq,255)
    return [X[lower_bound:upper_bound],frq[lower_bound:upper_bound]]
    #plt.show()
def main():
    for i in range(1,11):
        fft_sig('C:\Work\speech_recognition\sit\sample_{}.wav'.format(i))
        #fft_sig('C:\Work\speech_recognition\Week_2\samples_with_whistle\sample_{}.wav'.format(i))
    for i in range(1,11):
        #fft_sig('C:\Work\speech_recognition\Week_2\samples_without_whistle\sample_{}.wav'.format(i))
        fft_sig('C:\Work\speech_recognition\down\sample_{}.wav'.format(i))
if __name__ == "__main__":
    main()
