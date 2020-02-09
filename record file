import wave
import pyaudio
import keyboard
import time
def print_pressed_keys(e):
    global bool_space
    global bool_esc
    #если пробел нажат...
    if e.name=='space' and e.event_type=='down':
        bool_space=True    
    #если пробел нажат...
    if e.name=='esc' and e.event_type=='down':
        bool_esc=True     
    return  

def record(p,file):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = file

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []
    time=int(RATE / CHUNK * RECORD_SECONDS)
    for i in range(0, time):
        data = stream.read(CHUNK)
        frames.append(data)

        



    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return frames[0]

def found_n():
    n=1
    while True:
        try:
            open('C:\Work\speech_recognition\Week_1\samples\sample_{}.wav'.format(n),'r')
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
            sample = 'C:\Work\speech_recognition\Week_1\samples\sample_{}.wav'.format(n)
            #keyboard.add_hotkey('space', lambda: record(p,sample)) 
            record(p,sample)  
            n+=1
            bool_space=False
            
if __name__ == "__main__":
    main()
