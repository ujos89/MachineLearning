def make_dict(dic, address, words):
    tmp = []
    tmp.append(address)
    tmp.append(words)
    dic = dic.append(tmp)


f = open('trn_mono.txt','r')
dic = []
words = []
while True:
    line = f.readline()
    if not line: break
    line_tmp = line[:]

    if line_tmp.startswith('"'):
        address = line_tmp.replace('"','').replace('lab','txt')
    elif line_tmp.startswith('#'):
        pass
    elif (line_tmp.startswith('.')):
        make_dict(dic, address, words)
        words = []
    else:
        line_tmp = line_tmp.strip('\n')
        words.append(line_tmp)

f.close()

print(dic)