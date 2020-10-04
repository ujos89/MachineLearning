import pandas as pd

group_idx = []
for i in range(1,49):
    if(i < 10):
        exec('group_idx.append("MAD0{}")'.format(i))
    else:
        exec('group_idx.append("MAD{}")'.format(i))

#load data
target = pd.read_csv('../data/preprocessing/test.csv')
target = target.iloc[:,:1]

for idx in group_idx:
    exec('pred_{} = pd.read_csv("../data/prediction/pred_{}.csv")'.format(idx,idx))
    exec('pred_{} = pd.concat([target, pred_{}.iloc[:, 1:]], axis=1)'.format(idx,idx))
    exec('pred_{} = pred_{}.groupby("stock_code")["label"].mean().reset_index()'.format(idx,idx))
    exec('pred_{} = pred_{}.sort_values(["label"], ascending=[False]).reset_index(drop=True)'.format(idx,idx))
    exec('answer_{} = list(pred_{}.iloc[:3,0])'.format(idx,idx))
    exec('answer_{}.sort()'.format(idx))
    exec('answer_{}.insert(0, "{}")'.format(idx,idx))

answer_data = []
for idx in group_idx:
    exec('answer_data.append(answer_{})'.format(idx))

answer = pd.DataFrame(answer_data, columns=['그룹명','종목번호1','종목번호2','종목번호3'])
print(answer)

answer.to_csv("../data/answer.csv", index=False)
