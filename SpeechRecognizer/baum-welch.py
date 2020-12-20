import numpy as np
from readhmm import phone, phone_list
from readtrn import trn
from utterance import build_utterancehmm
from accumulate import *

phone_seq0, transp0, theta0 = build_utterancehmm(trn[0], phone_list, phone)
transp_, mu_, sigma_, gamma = accumulate(readdata(trn[0]), transp0, theta0)
