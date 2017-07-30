import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import cross_validation
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

#Load dataset
train_features = pd.read_csv('train_values.csv', index_col=0)
train_ori = pd.read_csv('train_values.csv', index_col=0)
train_label = pd.read_csv('train_labels.csv', index_col=0)
test_features = pd.read_csv('test_values.csv', index_col=0)
test_label = pd.read_csv('submission_format.csv',index_col=0)
#train = pd.concat([train_features, train_label], axis=1)

#Info of dataset
train_label.repayment_rate.describe()
#print(train_label.repayment_rate.median())


#School types and repayment rate
'''
train_profit = train_features.loc[train_features['school__ownership']=='Private for-profit']
train_nonprofit = train_features.loc[train_features['school__ownership']=='Private nonprofit']
train_public = train_features.loc[train_features['school__ownership']=='Public']
print('The median of for-profit school is '+str(train_profit['repayment_rate'].median()))
print('The median of non-profit school is '+str(train_nonprofit['repayment_rate'].median()))
print('The median of public school is '+str(train_public['repayment_rate'].median()))

#Correlation of repayment rate and SAT average
print('Correlation of repayment rate and Sat average is '+str(train_features['admissions__sat_scores_average_overall'].corr(train_features['repayment_rate'])))
print('Correlation of repayment rate and median family income is '+str(train_features['student__demographics_median_family_income'].corr(train_features['repayment_rate'])))


#Areas and repayment rate
train_greatlake = train_features.loc[train_features['school__region_id']=='Great Lakes (IL, IN, MI, OH, WI)']
train_farwest = train_features.loc[train_features['school__region_id']=='Far West (AK, CA, HI, NV, OR, WA)']
train_newengland = train_features.loc[train_features['school__region_id']=='New England (CT, ME, MA, NH, RI, VT)']
train_rockymountain = train_features.loc[train_features['school__region_id']=='Rocky Mountains (CO, ID, MT, UT, WY)']
print('The median repayment rate of Great Lake is '+str(train_greatlake['repayment_rate'].median()))
print('The median repayment rate of Far West is '+str(train_farwest['repayment_rate'].median()))
print('The median repayment rate of New England is '+str(train_newengland['repayment_rate'].median()))
print('The median repayment rate of Rocky Mountain is '+str(train_rockymountain['repayment_rate'].median()))
'''

'''
for i in idx:
    print('Column '+i+' has '+str(train_features[i].isnull().sum())+' NaNs.')
'''


#get info of dataset
idx = train_features.columns


def get_info(df):
    for i in range(len(idx)):
        print('Column '+str(i)+': '+idx[i]+' has '+str(df[idx[i]].isnull().sum())+' NaNs and '+str(len(df[idx[i]].unique()))+' unique values. Dtype is '+str(df[idx[i]].dtypes))


def get_nans(df):
    for i in range(len(idx)):
        nans = df[idx[i]].isnull().sum()
        nans_per = float(nans)/float(len(df[idx[i]]))
        if nans_per >= 0.2:
            print('Column '+str(i)+': '+idx[i]+' has '+str(nans)+' ('+str(nans_per)+')')
#get_info(test_features)



def print_data_type(df,target_type):
    for i in idx:
        if df[i].dtypes == target_type:
            print('Column '+str(i)+'         '+str(df[i].dtypes)+'       '+str(len(df[i].unique())))

        
def mapping_generate(df,col):
    mapp = {}
    ori = df[col].unique()
    for i in range(len(ori)):
        if pd.isnull(ori[i]) == False:
            mapp.update({ori[i]:int(i)})
    return mapp





def get_ori_info(df_ori):
    idx = df_ori.columns
    obj_idx = []
    drop_idx = []
    thres = 0.3
    for i in idx:
        nan_per = float(df_ori[i].isnull().sum())/float(len(df_ori[i]))
        if df_ori[i].dtypes == object:
            obj_idx.append(i)
        if nan_per >= thres:
            drop_idx.append(i)
    obj_idx_not_drop = []
    for i in obj_idx:
        if i not in drop_idx:
            obj_idx_not_drop.append(i)
    return obj_idx_not_drop, drop_idx
    






'''

def get_obj_idx_map(df):    
    col_names = []
    col_maps = []
    for i in idx:
        if df[i].dtypes == object:
            col_names.append(i)
            mapp = mapping_generate(df,i)
            col_maps.append(mapp)
    return col_names,col_maps

def obj2int(df,col_names,col_maps):
    for i in idx:
        if i in col_names:
            map_temp = col_maps[col_names.index(i)]
            df.replace({i:map_temp},inplace=True)



def get_drop_index(df,thres):
    drop_idx=[]
    for i in idx:
        nan_per = float(df[i].isnull().sum())/float(len(df[i]))
        if nan_per >= thres:
            drop_idx.append(i)
    return drop_idx
'''    

def fillnas(df):
    idx = train_features.columns
    
    col_set1 = idx[0:190]
    for col in col_set1:
        df[col] = df[col].fillna(0).astype(np.int64)
     
    col_set2 = idx[190:228]
    for col in col_set2:
        df[col] = df[col].fillna(df[col].mean())
     
    col_set3 = idx[228:253]
    for col in col_set3:
        df[col] = df[col].fillna(df[col].mean())
     
    col_set4 = idx[253:290]
    for col in col_set4:
        df[col] = df[col].fillna(df[col].mean())
     
    col_set5 = idx[290:322]
    for col in col_set5:
        df[col] = df[col].fillna(df[col].mean())
     
    col_set6 = idx[322:357]
    for col in col_set6:
        df[col] = df[col].fillna(df[col].mean())
     

     

     
    col_set9 = idx[364:366]
    for col in col_set9:
        df[col] = df[col].fillna(df[col].mean())
     

 
    col_set10 = idx[367:368]
    for col in col_set10:
        df[col] = df[col].fillna(df[col].mean())
 

     
    col_set13 = idx[383:384]
    for col in col_set13:
        df[col] = df[col].fillna(df[col].mean())
     

 
    col_set15 = idx[385:443]
    for col in col_set15:
        df[col] = df[col].fillna(df[col].mean())
 





obj_idx, drop_idx = get_ori_info(train_features)


fillnas(train_features)
fillnas(test_features)








train_features.drop(drop_idx, axis=1, inplace=True)
test_features.drop(drop_idx, axis=1, inplace=True)
train_dummy = pd.get_dummies(train_features[obj_idx])
test_dummy = pd.get_dummies(test_features[obj_idx])
train_dummy_missing = test_dummy.columns.difference(train_dummy.columns).tolist()
test_dummy_missing = train_dummy.columns.difference(test_dummy.columns).tolist()
train_features.drop(obj_idx, axis=1, inplace=True)
test_features.drop(obj_idx, axis=1, inplace=True)

for i in train_dummy_missing:
    train_dummy[i] = 0
for i in test_dummy_missing:
    test_dummy[i] = 0
train_dummy.sort_index(axis=1, inplace=True)
test_dummy.sort_index(axis=1, inplace=True)



train_features = pd.concat([train_features, train_dummy], axis=1)
test_features = pd.concat([test_features, test_dummy], axis=1)




#train_label['repayment_rate'] = pd.to_numeric(train_label['repayment_rate'])





'''
pca = PCA(n_components=30, svd_solver='full')
pca.fit(train_features)
train_features = pca.transform(train_features)
test_features = pca.transform(test_features)

'''







test_model = False


clf_type = 'randomforest'

if clf_type == 'linear':

    if test_model is True:
        train_label_p = train_label['repayment_rate'].copy()
        X_train, X_test, y_train, y_test = train_test_split(train_features, train_label_p, test_size=0.3, random_state=1)
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)
        
        #cross_validation.cross_val_score(model, X_train, y_train, cv=5).mean()
        
        
        predictions = model.predict(X_test)
        rms = sqrt(mean_squared_error(y_test, predictions))
        print(rms)
        
    else:
        train_label_p = train_label['repayment_rate'].copy()
        model = linear_model.LinearRegression()
        model.fit(train_features,train_label_p)
        pred = model.predict(test_features)

        for i in range(len(pred)):
            if pred[i] > 100:
                pred[i] = 100

        test_label['repayment_rate'] = pred
        test_label.to_csv('result.csv')
        
if clf_type == 'svm':
    if test_model == True:
        train_label_p = train_label['repayment_rate'].copy()
        X_train, X_test, y_train, y_test = train_test_split(train_features, train_label_p, test_size=0.4, random_state=4)
        model = svm.SVR(kernel='rbf')
        model.fit(X_train,y_train)
        cross_validation.cross_val_score(model, X_train, y_train, cv=5).mean()
        predictions = model.predict(X_test)
        rms = sqrt(mean_squared_error(y_test, predictions))
        print(rms)
    else:
        train_label_p = train_label['repayment_rate'].copy()
        model = svm.SVR(kernel='linear')
        model.fit(train_features,train_label_p)
        pred = model.predict(test_features)
        test_label['repayment_rate'] = pred
        test_label.to_csv('result.csv')
        
if clf_type == 'randomforest':

    if test_model is True:
        train_label_p = train_label['repayment_rate'].copy()
        X_train, X_test, y_train, y_test = train_test_split(train_features, train_label_p, test_size=0.3, random_state=1)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        #cross_validation.cross_val_score(model, X_train, y_train, cv=5).mean()
        
        
        predictions = model.predict(X_test)
        rms = sqrt(mean_squared_error(y_test, predictions))
        print(rms)
        
    else:
        train_label_p = train_label['repayment_rate'].copy()
        model = RandomForestRegressor()
        model.fit(train_features,train_label_p)
        pred = model.predict(test_features)
        
        test_label['repayment_rate'] = pred
        test_label.to_csv('result.csv')
      
