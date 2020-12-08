#import numpy as np
# structed array numpy 
def build_hmm_model(trans):
    hmm = 0
    return hmm


#read file 
f = open("hmm.txt", 'r')
hmm = []
hmm_list =[]

state2 = state3 = state4 = trans_sig=pdf_num = mu = sigma = 0,0,0,0,0
mu1, mu2, sigma1, sigma2, trans = [],[],[],[],[]
while True:
    line = f.readline()
    if not line: break
    
    line_tmp = line[:]
    print(line_tmp)

    if line_tmp.startswith('~h'):
        hmm_name = line_tmp.replace('~h ', '').replace('"','')
        hmm_list.append(hmm_name)
    elif (line_tmp == '<STATE> 2'):
        state2 = 1
    elif (line_tmp == '<STATE> 3'):
        state3 = 1
    elif (line_tmp == '<STATE> 4'):
        state4 = 1
    elif line_tmp.startwith('<TRANSP>'):
        trans_sig = int(line_tmp.replace('<TRANSP> ', ''))
    elif line_tmp.startswith('<MIXTURE>'):
        pdf_num = int(line_tmp.replace('<MIXUTRE> ', '').split('')[0])
        w = float(line_tmp.replace('<MIXUTRE> ', '').split('')[1])
    elif (line_tmp == '<EMDHMM>'):
        hmm.append(build_hmm_model(trans))
        #initalize
        state2, state3, state4, trans_sig, pdf_num = 0,0,0,0,0
        trans.clear()
        mu1.clear()
        mu2.clear()
        sigma1.clear()
        sigma2.clear()

    if(pdf_num == 1):
        if(line_tmp.stratswith('<MEAN>')):
            mu = 1
        elif(mu == 1):
            mu1.append(list(map(float,line_tmp.split(' '))))
            mu = 0
    elif(pdf_num == 2):
        if(line_tmp.stratswith('<MEAN>')):
            mu = 1
        elif(mu == 1):
            mu2.append(list(map(float,line_tmp.split(' '))))
            mu = 0

    #save transition probability matrix
    if(trans_sig < 5 and trans_sig > 0):
        trans_sig -= 1
        trans.append(line_tmp.split(' '))
        if(trans_sig == 0):
            trans.clear()
    
    
f.close()