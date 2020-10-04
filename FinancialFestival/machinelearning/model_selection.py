import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#load data
group_idx = []
for i in range(1,49):
    if(i < 10):
        exec('group_idx.append("MAD0{}")'.format(i))
    else:
        exec('group_idx.append("MAD{}")'.format(i))

train = []
label = []
test = pd.read_csv("../data/preprocessing/test.csv")
test = test.iloc[:,1:]
for idx in group_idx:
    exec('train_{} = pd.read_csv("../data/preprocessing/train/train_{}.csv")'.format(idx,idx))
    exec('label_{} = pd.read_csv("../data/preprocessing/label/label_{}.csv")'.format(idx,idx))
    exec('train.append(train_{}.iloc[:,2:])'.format(idx))
    exec('label.append(label_{}.iloc[:,1:])'.format(idx))
'''
#### EMERGENCY
lin_reg = LinearRegression()
lin_reg.fit(train[0],label[0])
print(len(test))

exit()
for i in range(len(test)):
    test_temp = np.array(test.iloc[i, :]).reshape(1,-1)
    print(lin_reg.predict(test_temp))
exit()
'''
#setting test set
test_set = []
for i in range(len(test)):
    test_temp = np.array(test.iloc[i, :]).reshape(1, -1)
    test_set.append(test_temp)

# linear regression, error: rmse
predict_lr = []
score_rmse_lr = []
for i, idx in enumerate(group_idx):
    exec('lin_reg{} = LinearRegression()'.format(idx))
    exec('lin_reg{}.fit(train[i],label[i])'.format(idx))
    predict_temp = []
    for test_idx in test_set:
        exec('pred_val = lin_reg{}.predict(test_idx)'.format(idx))
        predict_temp.append(float(pred_val))
    predict_lr.append(predict_temp)
    exec('score_rmse_lr.append( np.sqrt(-1 * cross_val_score(lin_reg{}, train[i], label[i], scoring="neg_mean_squared_error", cv=5)).mean() )'.format(idx))


# decisiontree regression, error: rmse
predict_tr = []
score_rmse_tr = []
for i, idx in enumerate(group_idx):
    exec('tree_reg{} = DecisionTreeRegressor()'.format(idx))
    exec('tree_reg{}.fit(train[i],label[i])'.format(idx))
    predict_temp = []
    for test_idx in test_set:
        exec('pred_val = tree_reg{}.predict(test_idx)'.format(idx))
        predict_temp.append(float(pred_val))
    predict_tr.append(predict_temp)
    exec('score_rmse_tr.append( np.sqrt(-1 * cross_val_score(tree_reg{}, train[i], label[i], scoring="neg_mean_squared_error", cv=5)).mean() )'.format(idx))

# randomforest regression, error: rmse
predict_fo = []
score_rmse_fo = []
for i, idx in enumerate(group_idx):
    exec('forest_reg{} = RandomForestRegressor()'.format(idx))
    train_temp = np.array(train[i])
    label_temp = np.array(label[i]).ravel()
    exec('forest_reg{}.fit(train_temp, label_temp)'.format(idx))
    predict_temp = []
    for test_idx in test_set:
        exec('pred_val = forest_reg{}.predict(test_idx)'.format(idx))
        predict_temp.append(float(pred_val))
    predict_fo.append(predict_temp)
    exec('score_rmse_fo.append( np.sqrt(-1 * cross_val_score(forest_reg{}, train_temp, label_temp, scoring="neg_mean_squared_error", cv=5)).mean() )'.format(idx))

#model selection
model = []
predict = [] 

for i, idx in enumerate(group_idx):
    min_val = min(score_rmse_lr[i], score_rmse_tr[i], score_rmse_fo[i]) 
    if ( min_val == score_rmse_lr[i] ):
        model.append('lr')
        predict.append(predict_lr[i])
    elif ( min_val == score_rmse_tr[i] ):
        model.append('tr')
        predict.append(predict_tr[i])
    else:
        model.append('fo')
        predict.append(predict_fo[i])
print(model)

#predction to datatable
for i, idx in enumerate(group_idx):
    exec('pred_{} = pd.DataFrame(predict[i], columns=["label"])'.format(idx))
    exec('pred_{}.to_csv("../data/prediction/pred_{}.csv")'.format(idx,idx))

