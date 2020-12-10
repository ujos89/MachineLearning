import numpy as np
from readhmm import phone, phone_list
from readtrn import trn

def build_utterancehmm(trn_utt, phone_list, phone):
    #ready for transp_seq, state_seq
    state_seq = []
    transp_seq = []
    for phone_hmm in trn_utt[1]:
        tmp_idx = phone_list.index(phone_hmm)
        transp_seq.append(tmp_idx)
        '''
        if(tmp_idx != 17):
            state_seq.append(tmp_idx)
            state_seq.append(tmp_idx)
            state_seq.append(tmp_idx)
        else:
            state_seq.append(tmp_idx)
        '''
    #state_seq = np.array(state_seq)
    transp_seq = np.array(transp_seq)
    #print(state_seq, type(state_seq), len(state_seq))
    #print(transp_seq, type(transp_seq), len(transp_seq))
 
    ## build theta
    #build transition probabilty
    before_transp_phone = -1
    for transp_phone in transp_seq:
        if(before_transp_phone != -1):
            transp_now = phone[transp_phone][-1]
            transp_len = len(transp)
            # + "sp"
            if(transp_phone == 17):
                transp_tmp = np.zeros((transp_len+1,transp_len+1))
                transp_tmp[:-1,:-1] = transp
                transp_tmp[-2,-2:] = transp_now[1,1:2]
                transp_tmp[-3,-2:] = transp_tmp[-3,-2] * transp_now[0,1:2]
                transp = transp_tmp
            # + other phone
            else:
                transp_tmp = np.zeros((transp_len+3,transp_len+3))
                transp_tmp[0:transp_len-1, 0:transp_len] = transp[0:transp_len-1, :]
                transp_tmp[-4:,-4:] = transp_now[-4:,-4:]
                transp = transp_tmp
        else:
            transp = phone[transp_phone][-1]
        before_transp_phone = transp_phone
    #print(transp, type(transp), len(transp))

    #build weight, mu, sigma
    states = []
    for idx in transp_seq:
        if(idx == 17):
            states.append(phone[idx][0])
        else:
            states.append(phone[idx][0])
            states.append(phone[idx][1])
            states.append(phone[idx][2])
    #state => staes[staes][pdf][w, mu, sigma]

    return transp_seq, transp, states

#how to use
phone_seq, transp, theta = build_utterancehmm(trn[0], phone_list, phone)
