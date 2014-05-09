from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import numpy

(rate,sig) = wav.read("affirmative.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)
fbank_feat2 = logfbank(sig,rate)

print type(fbank_feat)
print fbank_feat.size
print fbank_feat.ndim
print fbank_feat.shape



this = numpy.concatenate([fbank_feat, fbank_feat2], axis=0)
print this
