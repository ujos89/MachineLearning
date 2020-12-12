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
#print(backward(readdata(trn[0]), transp0, theta0))
print(len(theta0[0]))

def accumulate(data, transp, theta):
    alpha, p = forward(data, transp, theta)
    beta = backward(data, transp, theta)

    time_num = data.shape[0]
    state_num = len(theta)
    #assume that pdf_num is same for each states
    pdf_num = len(theta[0])
    transp_ = np.zeros((state_num+2, state_num+2))
    mu_ = [[np.zeros(39) for _ in range(pdf_num)] for _ in range(state_num)]
    sigma_ = [[np.zeros(39) for _ in range(pdf_num)] for _ in range(state_num)]
    gamma = np.zeros((state_num,pdf_num+1))
    
    for time_idx in range(time_num):
        for state_idxi in range(state_num):
            gamma_tmp0 = alpha[state_idxi][time_idx]+beta[state_idxi][time_idx]-p
            gamma[state_idxi][0] += np.exp(gamma_tmp0)
            if(time_idx == 0):
                transp_[0][state_idxi+1] += np.exp(gamma_tmp0)
            elif(time_idx != time_num-1):
                for state_idxj in range(state_num):
                    xi_tmp = alpha[state_idxi][time_idx] + transp[state_idxi+1][state_idxj] + obsp_i(data[time_idx+1], theta[state_idxj]) + beta[state_idxj][time_idx+1] - p
                    transp_[state_idxi+1][state_idxj+1] += np.exp(xi_tmp)
            for pdf_idx in range(1,pdf_num+1):
                theta_tmp = []
                w = theta[pdf_idx][0]
                mu = theta[pdf_idx][1]
                sigma = theta[pdf_idx][2]
                x = data[time_idx]
                obsp_ik = np.log(w)-np.log(((2*math.pi)**(39/2))*(np.prod(sigma)**0.5))-0.5*np.sum((x-mu)**2/sigma)
                
                gamma_tmp = np.exp(gamma_tmp0 + w + obsp_ik - obsp_i(data[time_idx], theta[state_idxi]))
                gamma[state_idxi][pdf_idx] += gamma_tmp
                mu_[state_idx][pdf_idx-1] += gamma_tmp*data[time_idx]
                sigma_[state_idx][pdf_idx-1] += gamma_tmp*(data[time_idx]**2)
    
    return transp_, mu_, sigma_, gamma

#need to verify accumulate
