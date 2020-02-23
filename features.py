import numpy as np
from python_speech_features import mfcc,ssc
from python_speech_features import logfbank,fbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def mfcc_loop(n_first,n_last,grade):
    a=[]
    b=[]
    for i in range(n_first, n_last):
        (rate,sig) = wav.read("C:\Work\speech_recognition\{}\sample_{}.wav".format(grade,i))
        a.append([i for i in ssc(sig,rate,nfft=1103)])
        b.append([i for i in logfbank(sig,rate,nfft=1103)])
    return a,b
sit= mfcc_loop(1,11,'sit')
down= mfcc_loop(1,11,'down')

logfbank_sit=np.array(sit[0])
mfcc_sit=np.array(sit[1])

logfbank_down=np.array(down[0])
mfcc_down=np.array(down[1])

x,y,z=logfbank_sit.shape[0],logfbank_sit.shape[1],logfbank_sit.shape[2]
logfbank_sit=logfbank_sit.reshape(x*y*z)

x,y,z=logfbank_down.shape[0],logfbank_down.shape[1],logfbank_down.shape[2]
logfbank_down=logfbank_down.reshape(x*y*z)

x,y,z=mfcc_sit.shape[0],mfcc_sit.shape[1],mfcc_sit.shape[2]
mfcc_sit=mfcc_sit.reshape(x*y*z)

x,y,z=mfcc_down.shape[0],mfcc_down.shape[1],mfcc_down.shape[2]
mfcc_down=mfcc_down.reshape(x*y*z)
#mfcc_sit=list(mfcc_sit)
#mfcc_sit+=[0]*77740
#mfcc_down=list(mfcc_down)
#mfcc_down+=[0]*77740
plt.scatter(logfbank_sit, mfcc_sit,label='sit')
plt.scatter(logfbank_down, mfcc_down,label='down')
plt.legend()
plt.show()
