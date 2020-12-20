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

def logplus(lna_, lnb_):
    lna, lnb = lna_, lnb_
    
    #inital state
    if lna == 0:
        return lnb
    elif lnb == 0:
        return lna
    
    #prevent overflow for expotential
    if (lna < lnb):
        lna, lnb = lnb, lna
    
    #ln(a+b) = lna + ln(1+e^(lnb-lna))
    return lna + np.log(1+np.exp(lnb-lna))
    
def obsp(x, theta_state):
    #calculate observation probability by w,mu,sigma
    pdf_num = len(theta_state)
    obsp = np.zeros(1)
    
    for idx in range(pdf_num):
        w = np.log(theta_state[idx][0])
        mu = theta_state[idx][1]
        sigma = theta_state[idx][2]
        obsp_tmp = w - (39/2*np.log(2*math.pi)) - 0.5*np.prod(sigma) - 0.5*np.sum((x-mu)**2/sigma)
        obsp = logplus(obsp, obsp_tmp)

    return obsp
#guide: x<- data[time_idx], theta_state<- theta[state_idx]

def forward(data, transp, theta):
    state_num = len(theta)
    time_num = data.shape[0]
    alpha = np.zeros((state_num, time_num))

    #state 0-> state 1 : probability 1, ln(1)=0
    alpha[0][0] = obsp(data[0], theta[0])

    for time_idx in range(1, time_num):
        for state_idxj in range(state_num):
            alpha_tmp = np.zeros(1)
            
            for state_idxi in range(state_num):
                transp_ij = transp[state_idxi+1][state_idxj+1]
                alpha_it = alpha[state_idxi][time_idx-1]
                if (transp_ij and alpha_it):
                    alpha_tmp = logplus(alpha_tmp, alpha_it+np.log(transp_ij))

            if alpha_tmp:
                alpha[state_idxj][time_idx] = alpha_tmp + obsp(data[time_idx] ,theta[state_idxj])

    #end condition only at last state
    p = alpha[-1][-1] + np.log(transp[-2][-1])
    return alpha, p

def backward(data, transp, theta):
    state_num = len(theta)
    time_num = data.shape[0]
    beta = np.zeros((state_num, time_num))
    
    #last state only have propertity 
    beta[-1][-1] = np.log(transp[-2][-1]) 

    for time_idx in reversed(range(time_num-1)):
        for state_idxi in range(state_num):
            beta_tmp = np.zeros(1)
            for state_idxj in range(state_num):
                transp_ij = transp[state_idxi+1][state_idxj+1]
                beta_jt = beta[state_idxj][time_idx+1]
                if (transp_ij and beta_jt):
                    obsp_j = obsp(data[time_idx+1], theta[state_idxj])
                    beta_tmp = logplus(beta_tmp, np.log(transp_ij)+obsp_j+beta_jt)
            
            beta[state_idxi][time_idx] = beta_tmp
    
    return beta

phone_seq0, transp0, theta0 = build_utterancehmm(trn[0], phone_list, phone)
#alpha_, p_ = forward(readdata(trn[0]), transp0, theta0)
#beta_ = backward(readdata(trn[0]), transp0, theta0)
'''
print(alpha_)
for _ in np.transpose(np.nonzero(alpha_)):
    print(_)
for _ in np.argwhere(alpha_>0):
print(_)
print(p_)
for _ in beta_:
    print(_)
'''
# reduce error from floating point
def get_gamma(a, b, p):
    if a==0 or b==0:
        return np.zeros(1)

    if a<b:
        a,b = b,a

    #define b-p > epsilon
    if ((b-p)/b < np.array([10**-10])):
        return a
    else:
        return a+b-p


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
    #gamma:log scale
    gamma = np.zeros((state_num,pdf_num+1))

    for time_idx in range(time_num):
        for state_idxi in range(state_num):
            gamma_tmp0 = get_gamma(alpha[state_idxi][time_idx], beta[state_idxi][time_idx], p)

            '''
            #floating point problem
            gamma_tmp0 = np.zeros(1)
            alpha_it = alpha[state_idxi][time_idx]
            beta_it = beta[state_idxi][time_idx]
            if(alpha_it and beta_it):
                gamma_tmp0_ = alpha_it + beta_it - p
                if(gamma_tmp0_<= 0):
                    gamma_tmp0 = gamma_tmp0_
                else:
                    print(gamma_tmp0_, " time:",time_idx," state:",state_idxi)
                    print(alpha_it, beta_it, p)
            '''
            gamma[state_idxi][0] = logplus(gamma[state_idxi][0], gamma_tmp0)
            
            if(time_idx == 0):
                #state0 -> state1: only case for these model
                transp_[0][1] = 1 
                
            elif(time_idx != time_num-1):
                for state_idxj in range(state_num):
                    transp_xi = transp[state_idxi+1][state_idxj+1]
                    if transp_xi:
                        transp_xi = np.log(transp_xi)
                        obsp_xi = obsp(data[time_idx+1],theta[state_idxj])
                        #print(transp_xi, obsp_xi, alpha[state_idxi][time_idx], beta[state_idxj][time_idx+1], p)
                        xi_tmp = get_gamma(alpha[state_idxi][time_idx], beta[state_idxj][time_idx+1], p)
                        xi_tmp = np.exp(xi_tmp+transp_xi + obsp_xi)
                        #print("*",xi_tmp," time:",time_idx," statei:",state_idxi," statej:",state_idxj)
                    
                    transp_[state_idxi+1][state_idxj+1] += xi_tmp

                    '''
                    xi_tmp = np.zeros(1)
                    alpha_xi = alpha[state_idxi][time_idx]
                    beta_xi = beta[state_idxj][time_idx+1]
                    transp_xi = transp[state_idxi+1][state_idxj+1]
                    if(alpha_xi and beta_xi and transp_xi):
                        obsp_xi = obsp(data[time_idx+1],theta[state_idxj])
                        xi_tmp = alpha_xi + np.log(transp_xi) + obsp_xi + beta_xi - p
                    
                    if (xi_tmp < 0):
                        transp_[state_idxi+1][state_idxj+1] += np.exp(xi_tmp)
                    elif (xi_tmp>0):
                        print(xi_tmp, state_idxi, state_idxj, time_idx)
                    
                    
                    transp_ij = transp_[state_idxi+1][state_idxj+1]
                    if(transp_ij):
                        transp_[state_idxi+1][state_idxj+1] = np.exp(logplus(np.log(transp_ij), xi_tmp))
                    elif(xi_tmp):
                        transp_[state_idxi+1][state_idxj+1] = np.exp(xi_tmp)
                    '''
            for pdf_idx in range(1,pdf_num+1):
                w = np.log(theta[state_idxi][pdf_idx-1][0])
                mu = theta[state_idxi][pdf_idx-1][1]
                sigma = theta[state_idxi][pdf_idx-1][2]
                x = data[time_idx]
                obsp_ik = w - (39/2*np.log(2*math.pi)) - 0.5*np.prod(sigma) - 0.5*np.sum((x-mu)**2/sigma)
                
                if gamma_tmp0:
                    gamma_tmp = gamma_tmp0 + w + obsp_ik - obsp(data[time_idx], theta[state_idxi])
                    gamma[state_idxi][pdf_idx] = logplus(gamma[state_idxi][pdf_idx] ,gamma_tmp)
                    mu_[state_idxi][pdf_idx-1] += np.exp(gamma_tmp)*data[time_idx]
                    sigma_[state_idxi][pdf_idx-1] += np.exp(gamma_tmp)*(data[time_idx]**2)

    return transp_, mu_, sigma_, gamma

transp_, mu_, sigma_, gamma = accumulate(readdata(trn[0]), transp0, theta0)
#print(transp_)
#print(np.transpose(np.nonzero(transp_)))
#print(transp_.sum(axis=1))