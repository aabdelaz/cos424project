import os, numpy
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav

# read in files
id = 0

# get speakers
# speakers = os.walk('./PDA').next()[1]
speakers = os.walk('./pdatest').next()[1]

dataList = []

# for each speaker
for speaker in speakers:
    for root, dirnames, filenames in os.walk('./pdatest/' + speaker): 
# os.walk('./PDA/' + speaker):
        for filename in filenames:
            (rate,sig) = wav.read(root + '/' + filename)
            mfcc_feat = mfcc(sig,rate)
            fbank_feat = logfbank(sig,rate)
            dataList.append(fbank_feat)

            print 'hi'

data = numpy.concatenate(dataList, axis=0)



# make copy

# train k svms

