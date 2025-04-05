# -*- coding: gbk -*-
import pandas as pd
import numpy as np
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split #���ݼ�����
from sklearn.ensemble import RandomForestClassifier # ���ɭ��
from sklearn.metrics import confusion_matrix, classification_report #����
from sklearn.tree import DecisionTreeClassifier #������
from sklearn.linear_model import LogisticRegression #�߼��ع�
from sklearn.metrics import accuracy_score #��ȷ��

def data_processing():
    data = pd.read_csv("C:/Users/yizhuo/Desktop/school_resource/��������ҵ����/ʵѵ��Ŀ3.ͨѶ��Ӫ�̿ͻ���ʧԤ��/USER_INFO_M.csv", encoding='gbk')

    print("1 data na status:", data.isnull().sum(), data.shape)

    ## 1.����ȥ��
    print("2 duplicating data:", data.duplicated().sum())  # �鿴�ظ�����  96

    data.drop_duplicates(inplace=True)  # ����ȥ��
    print("3 result", data.duplicated().sum())#0--�ɹ�

    ## 2.��ά
    data.drop(['MANU_NAME', 'MODEL_NAME', 'OS_DESC', 'CONSTELLATION_DESC'], axis=1, inplace=True)

    print("4 result data columns:", data.columns)
    cleardata = data

    ## ���ݺϲ���ͨ��ID��
    data_group = cleardata.groupby('USER_ID')  # ����

    ## ��������
    label = data_group[['USER_ID', 'IS_LOST']].tail(1)  # ȡ�û�id����ǣ�ÿ������һ��ֵ��
    label.set_index('USER_ID', inplace=True)  # ����USER_ID����Ϊ����
    print(label)


    # 3.2 ȥ�أ��ϲ��Ա����䣬�ն����͡�ȡ��һ��
    data_1 = data_group[['CUST_SEX', 'CERT_AGE', 'TERM_TYPE','']].first() 
    print("data 5.1\n",data_1)

    # 3.3 ȥ�أ�����ʱ����ѡ�����һ�ε�����
    data_2 = data_group['INNET_MONTH'].last()
    print("data 5.2\n",data_2)



    ##���ݺϲ�

    ##��Լ��Ч���
    def cal_is_agree(x):  # x Ϊÿ���û���������ֵ
        # ��������²�ȫΪ1���õ�������ֵ��ȥǰ�����¾�ֵ�������µ�ֵ��Ϊ1��ȡֵΪ1.5��
        # ����ȡֵ���Ϊ-1��-0.5��0��0.5��1��1.5
        x = np.array(x)
        if x.sum() == 3:
            return 1.5
        else:
            return x[2] - x[:2].mean()

    data_3 = pd.DataFrame(data_group['IS_AGREE'].agg(cal_is_agree))#agg��һ���ۺϺ������ۺϺ�������ʼ�������ᣨĬ�������ᣬҲ���������ᣩ��ִ�У�
    print("data 5.3\n",data_3)

    ##����ȱʧֵ
    # 3.5 insert info into the label('AGREE_EXP_DATE') agree date
    date = data_group['AGREE_EXP_DATE'].last()  # ȡ��3���µ�"��Լ�ƻ�����ʱ��"
    num_mon = (pd.to_datetime(date, format='%Y%m') - pd.to_datetime('2016-03')).dt.days/30  # ʱ���ԡ��¡�Ϊ��λ
    data_4 = pd.DataFrame(num_mon).fillna(-1)    #��-1���ȱʧֵ

    print("data 5.4\n",data_4)


    #*********************************************************************************************************************************************
    # 3.6 insert info into the label('CREDIT_LEVEL') level
    data_5 = pd.DataFrame(data_group['CREDIT_LEVEL'].agg('mean'))    # ���õȼ�
    print("data 5.5\n",data_5)


    # 3.7 ȱʧֵ-VIP�ȼ�
    data_6 = data_group['VIP_LVL'].last().fillna(0)    # ȡ���һ��ֵ
    print("data 5.6\n",data_6)

    # 3.8 ���·���(ȡ�����µ�ƽ��ֵ)��������
    data_7 = pd.DataFrame(data_group['ACCT_FEE'].mean())
    print("data 5.7\n",data_7)


    # 3.9 ƽ��ÿ��ͨ��ʱ��
    # ��ͨ��
    data_8_1 = pd.DataFrame(data_group['CALL_DURA'].sum()/data_group['CDR_NUM'].sum(),
                            columns=['Total_mean'])
    # ����ͨ��
    data_8_2 = pd.DataFrame(data_group['NO_ROAM_LOCAL_CALL_DURA'].sum()/data_group['NO_ROAM_LOCAL_CDR_NUM'].sum(),
                            columns=['Local_mean'])
    # ���ڳ�;ͨ��
    data_8_3 = pd.DataFrame(data_group['NO_ROAM_GN_LONG_CALL_DURA'].sum() / data_group['NO_ROAM_GN_LONG_CDR_NUM'].sum(),
                            columns=['GN_Long_mean'])
    # ��������ͨ��
    data_8_4 = pd.DataFrame(data_group['GN_ROAM_CALL_DURA'].sum() / data_group['GN_ROAM_CDR_NUM'].sum(),
                            columns=['GN_Roam_mean'])
    # ����ƴ��
    data_8 = pd.concat([data_8_1, data_8_2, data_8_3, data_8_4], axis=1).fillna(0)

    print("data 5.8\n",data_8.head())


    # 3.10 ��������
    # ������ͨ���������Σ������ŷ���������������������(MB)�����ط�������������(MB)������������������(MB)��
    # ��ͨ���������������������б�������  ������ + ���� �� ��ͨ����
    # ��������Ȧ�����к���Ȧ�����к���Ȧ
    data_9 = data_group[['NO_ROAM_CDR_NUM', 'P2P_SMS_CNT_UP', 'TOTAL_FLUX', 'LOCAL_FLUX','GN_ROAM_FLUX',
                        'CALL_DAYS', 'CALLING_DAYS', 'CALLED_DAYS',
                        'CALL_RING','CALLING_RING', 'CALLED_RING']].agg('mean')
    print("data 5.9\n",data_9)


    #*********************************************************************************************************************************************
    # ����������&��ǩ���������������Ա�֤����ƴ��ʱ����һ��
    label.sort_index(inplace=True)
    data_1.sort_index(inplace=True)
    data_2.sort_index(inplace=True)
    data_3.sort_index(inplace=True)
    data_4.sort_index(inplace=True)
    data_5.sort_index(inplace=True)
    data_6.sort_index(inplace=True)
    data_7.sort_index(inplace=True)
    data_8.sort_index(inplace=True)
    data_9.sort_index(inplace=True)
    # ƴ����������&���
    data_new = pd.concat([data_1, data_2, data_3, data_4,
            data_5, data_6, data_7, data_8, data_9, label], axis=1)
    data_new.head()

    # 4 drop nall data
    #ȱʧֵ����

    print("6 isnull \n",data_new.isnull().sum())    # �鿴ȱʧֵ
    data_new = data_new.fillna(method='ffill').fillna(method='bfill')      # ����ֵ���(�������+�������)

    data_new.to_csv('C:/Users/yizhuo/Desktop/school_resource/��������ҵ����/ʵѵ��Ŀ3.ͨѶ��Ӫ�̿ͻ���ʧԤ��/clear_data.csv', index=True, encoding='utf-8-sig')


data = pd.read_csv('C:/Users/yizhuo/Desktop/school_resource/��������ҵ����/ʵѵ��Ŀ3.ͨѶ��Ӫ�̿ͻ���ʧԤ��/clear_data.csv', index_col=0)
corr = data.corr()    # Ƥ��ѷ���ϵ��    ɸѡ��ͻ���ʧ������ص�����
corr
print(data['IS_LOST'])



##############################################################################################################

## ��ȡ��������
# ��0.08��Ϊɸѡ��ֵ
feature_index = corr['IS_LOST'].drop('IS_LOST').abs() > 0.08    # ȡ����"���"�����ϵ��
feature_name = feature_index.loc[feature_index].index           # ѡ������Ҫ������
print(feature_name)


X = data.loc[:, feature_name]    # �����Ա���
y = data.loc[:, 'IS_LOST']       # ����Ŀ�����
# ������ƽ�� 
y.value_counts()


index_positive = y.index[y == 1]          # ������������
index_negative = np.random.choice(a=y.index[y == 0].tolist(), size=y.value_counts()[1])   # ���������������Ը����������²�������

X_positive = X.loc[index_positive, :]     # �������Ա���
X_negative = X.loc[index_negative, :]     # �������Ա���

y_positive = y.loc[index_positive]        # ��������ǩ
y_negative = y.loc[index_negative]        # ��������ǩ

X = pd.concat([X_positive, X_negative], axis=0)    # ������������
y = pd.concat([y_positive, y_negative], axis=0)    # �����ĸ�����

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # ����y�ı������ֲ����

# def Data_visualization():

##ģ��-���ɭ��
def RandomForest():
    rfc = RandomForestClassifier()    # ��ʼ�����ɭ��ģ��

    rfc.fit(X_train, y_train)         # ģ��ѵ��

    y_pre = rfc.predict(X_test)       # ����ģ�ͶԲ�����������Ԥ��
    print(classification_report(y_test, y_pre))    # ��ӡ���౨�棨�����˸�ģ����������ָ�꣩
    lr_acc = round(accuracy_score(y_pre,y_test)*100,2)
    print(f"logistic accuracy is: {lr_acc}%")


##ģ��--������
def DecisionTree():
    # ����������ģ��
    dtc = DecisionTreeClassifier()

    # ѵ��ģ��
    dtc.fit(X_train,y_train)
    # Ԥ��ѵ�����Ͳ��Լ����
    dtc_pred = dtc.predict(X_test)

    # ���㾫ȷ��
    dtc_acc = round(accuracy_score(dtc_pred,y_test)*100,2)
    print(f"decision tree accuracy is: {dtc_acc}%")

#ģ��-�߼��ع�
def Logistic():
    lr = LogisticRegression()

    # ѵ��ģ��
    lr.fit(X_train,y_train)

    # Ԥ��ѵ�����Ͳ��Լ����
    lr_pred = lr.predict(X_test)

    # ���㾫ȷ��
    lr_acc = round(accuracy_score(lr_pred,y_test)*100,2)
    print(f"logistic accuracy is: {lr_acc}%")
def depth():
    X_train1= np.array(X_train)
    y_train1 = np.array(y_train)
    X_test1 = np.array(X_test)
    y_test1 = np.array(y_test)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train1)
    X_test_scaled = scaler.transform(X_test1)

    # �������ѧϰģ��
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train1.shape[1],)),
        # layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # ����ģ��
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ѵ��ģ��
    model.fit(X_train_scaled, y_train1, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test1))


    
if __name__=='__main__':
    # data_processing()
    # Data_visualization()
    # RandomForest()    #78.25%
    # DecisionTree()  #69.83%
    depth()      #0.7905
    # Logistic()     #77.45%