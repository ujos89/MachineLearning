import numpy as np
from readhmm import phone, phone_list
from readtrn import trn

def build_utterancehmm(trn_utt, phone_list, phone):
    #ready for seqence of phone
    phone_seq = []
    for phone_hmm in trn_utt[1]:
        phone_seq.append(phone_list.index(phone_hmm))
    phone_seq = np.array(phone_seq)

    #build transition probabilty
    first_sig = 0
    for transp_phone in phone_seq:
        if first_sig:
            transp_now = phone[transp_phone][-1]
            transp_len = len(transp)
            # + "sp"
            if(transp_phone == 17):
                transp_tmp = np.zeros((transp_len+1,transp_len+1))
                transp_tmp[:-2,:-2] = transp[:-1,:-1]
                transp_tmp[-2,-2:] = transp_now[1,1:]
                transp_tmp[-3,-2:] = transp[-2,-1] * transp_now[0,1:]
                transp = transp_tmp
            # + other phone
            else:
                transp_tmp = np.zeros((transp_len+3,transp_len+3))
                transp_tmp[0:transp_len-1, 0:transp_len] = transp[:-1, :]
                transp_tmp[-4:,-4:] = transp_now[-4:,-4:]
                transp = transp_tmp
        else:
            transp = phone[transp_phone][-1]
            first_sig = 1

    #print(transp, type(transp), len(transp))

    #build weight, mu, sigma
    states = []
    for idx in phone_seq:
        if(idx == 17):
            states.append(phone[idx][0])
        else:
            states.append(phone[idx][0])
            states.append(phone[idx][1])
            states.append(phone[idx][2])
    #state => staes[staes][pdf][w, mu, sigma]

    return phone_seq, transp, states

'''
#how to use
phone_seq, transp, theta = build_utterancehmm(trn[0], phone_list, phone)
#print(transp.sum(axis=1))
#print(phone[20][-1])
#print(phone[20][-1].sum(axis=1))
'''