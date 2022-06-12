from asyncio.windows_events import NULL
from mel_features import *

from bark_features import *

from gammatone_features import *


def extract_features(sig,rate):

    mfcc_feat,band_eng, total_eng=mfcc(sig,rate)

    


    gfcc_feat,gfcc_energy=gfcc(sig=sig,rate=rate)




    bfcc_feat=bfcc(sig=sig,rate=rate)




    return mfcc_feat,band_eng, total_eng,gfcc_feat,bfcc_feat,gfcc_energy

