def make_trn(trn, address, words):
    tmp = []
    tmp.append(address)
    tmp.append(words)
    trn = trn.append(tmp)


f = open('trn_mono.txt','r')
trn = []
words = []
while True:
    line = f.readline()
    if not line: break
    line_tmp = line[:].strip()

    if line_tmp.startswith('"'):
        address = line_tmp.replace('"','').replace('lab','txt')
    elif line_tmp.startswith('#'):
        pass
    elif (line_tmp == '.'):
        make_trn(trn, address, words)
        words = []
    else:
        line_tmp = line_tmp.strip()
        words.append(line_tmp)

f.close()