import numpy as np

# read file 
f = open("hmm.txt", 'r')
phone_list = []
phone = []

# initalize signals for text mining 
hmm_sig, mu_sig, sigma_sig, pdf_sig, state_sig, transp_sig = 0,0,0,0,0,0
transp = []
pdf = []
state = []
hmm = []

while True:
    line = f.readline()
    if not line: break
    line_tmp = line[:].strip()

    if line_tmp.startswith('~h'):
        hmm_sig = 1
        name = line_tmp.replace('~h ','').replace('"','')
        phone_list.append(name)
        phone_sig = 1

    if (hmm_sig == 1):
        if line_tmp.startswith('<NUMSTATES>'):
            state_sig = int(line_tmp[-1]) - 1
        elif line_tmp.startswith('<STATE>'):
            state_sig -= 1
        elif line_tmp.startswith('<NUMMIXES>'):
            pdf_sig = int(line_tmp[-1]) + 1
        elif line_tmp.startswith('<MIXTURE>'):
            w = np.array(float(line_tmp.split()[-1]))
            pdf.append(w)
        elif line_tmp.startswith('<MEAN>'):
            mu_sig = 1
        elif line_tmp.startswith('<VARIANCE>'):
            sigma_sig = 1
        elif line_tmp.startswith('<TRANSP>'):
            transp_sig = int(line_tmp[-1])
        elif line_tmp.startswith('<ENDHMM>'):
            hmm_sig = 0
            transp = np.array(transp)
            hmm.append(transp)
            transp = []
            phone.append(hmm)
            hmm = [] 

        if(mu_sig and (not line_tmp.startswith('<MEAN>'))):
            mu_sig -= 1
            mu = np.array([float(x) for x in line_tmp.split()])
            pdf.append(mu)
        if(sigma_sig and (not line_tmp.startswith('<VARIANCE>'))):
            sigma_sig -= 1
            sigma = np.array([float(x) for x in line_tmp.split()])
            pdf.append(sigma)
            pdf_sig -= 1
            state.append(pdf)
            pdf = []
        if(pdf_sig == 1):
            pdf_sig -= 1
            hmm.append(state)
            state = []
        if(transp_sig and (not line_tmp.startswith('<TRANSP>'))):
            transp_sig -= 1
            transp_tmp = [float(x) for x in line_tmp.split()]
            transp.append(transp_tmp)

f.close()