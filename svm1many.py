import os
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav

# read in files
id = 0

# get speakers
speakers = os.walk('./audio_data').next()[1]

# for each speaker
for speaker in speakers:
    for root, dirnames, filenames in os.walk('./audio_data/' + speaker):
        for filename in filenames:
            
    id += 1

(rate,sig) = wav.read("affirmative.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print fbank_feat


    # find mfcc transformation

    # add tag

# make copy

# train k svms

