import numpy as np
import math
from readhmm import phone, phone_list
from readtrn import trn
from utterance import build_utterancehmm

def readdata(trn_idx):
    addr = trn_idx[0]
    f = open(addr, 'r')
    data_num = int(f.readline().strip().split()[0])
    idx = 0
    data = []
    while (idx < data_num):
        data_tmp = np.array([float(x) for x in f.readline().strip().split()])
        data.append(data_tmp)
        idx += 1
    f.close()
    data = np.array(data)
    return data

def obsp_i(x, theta_state):
    #calculate observation probability
    pdf_num = len(theta_state)
    obsp_i = np.zeros(1)
    for idx in range(pdf_num):
        w = theta_state[idx][0]
        mu = theta_state[idx][1]
        sigma = theta_state[idx][2]
        #ln(a+b)=lna+lnb for very small a, b
        obsp_i += np.log(w)-np.log(((2*math.pi)**(39/2))*(np.prod(sigma)**0.5))-0.5*np.sum((x-mu)**2/sigma)

    return obsp_i

def forward(data, transp, theta):
    state_num = len(theta)
    time_num = data.shape[0]
    alpha = np.zeros((state_num, time_num))
    #state 0-> state 1 : probability 1, ln(1)=0
    alpha[0][0] = obsp_i(data[0], theta[0])

    for time_idx in range(1, time_num):
        for state_idxj in range(state_num):
            alpha_tmp = np.zeros(1)
            for state_idxi in range(state_num):
                transp_ij = transp[state_idxi+1][state_idxj+1]
                if (transp_ij):
                    alpha_it = alpha[state_idxi][time_idx-1]
                    if(alpha_it):
                        alpha_tmp += alpha_it + np.log(transp_ij)
            alpha[state_idxj][time_idx] = alpha_tmp + obsp_i(data[time_idx] ,theta[state_idxj])

    #end condition only at last state
    p = alpha[-1][-1] + np.log(transp[state_num][-1])
    return alpha, p

def backward(data, transp, theta):
    state_num = len(theta)
    time_num = data.shape[0]
    beta = np.zeros((state_num, time_num))
    #last state only have propertity 
    beta[state_num-1][time_num-1] = np.log(transp[-2][-1]) 

    for time_idx in reversed(range(time_num-1)):
        for state_idxi in range(state_num):
            beta_tmp = np.zeros(1)
            for state_idxj in range(state_num):
                transp_ij = transp[state_idxi+1][state_idxj+1]
                if(transp_ij):
                    beta_jt = beta[state_idxj][time_idx+1]
                    if(beta_jt):
                        obsp_j = obsp_i(data[time_idx+1], theta[state_idxj])
                        beta_tmp += np.log(transp_ij) + obsp_j + beta_jt
            beta[state_idxi][time_idx] = beta_tmp
    
    return beta
    

phone_seq0, transp0, theta0 = build_utterancehmm(trn[0], phone_list, phone)
#print(forward(readdata(trn[0]), transp0, theta0))
print(backward(readdata(trn[0]), transp0, theta0))
