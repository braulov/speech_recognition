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
    t = np.arange(len(y)) / float(sr)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')
    
    frq, X = frequency_sepectrum(y, sr)
    
    
    start=time.time()
    a=quick_search(frq,5000)
    finish=time.time()
    if define_whistle(list(X),a):
        print("It's a whistle")
    else:
        print("It isn't a whistle")
    plt.subplot(2, 1, 2)
    plt.plot(frq, X, 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|X(freq)|')
    plt.tight_layout()
    return [X,frq]
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
