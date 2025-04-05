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
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.metrics import confusion_matrix, classification_report #报告
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.metrics import accuracy_score #精确度

def data_processing():
    data = pd.read_csv("C:/Users/yizhuo/Desktop/school_resource/互联网商业智能/实训项目3.通讯运营商客户流失预测/USER_INFO_M.csv", encoding='gbk')

    print("1 data na status:", data.isnull().sum(), data.shape)

    ## 1.数据去重
    print("2 duplicating data:", data.duplicated().sum())  # 查看重复数据  96

    data.drop_duplicates(inplace=True)  # 数据去重
    print("3 result", data.duplicated().sum())#0--成功

    ## 2.降维
    data.drop(['MANU_NAME', 'MODEL_NAME', 'OS_DESC', 'CONSTELLATION_DESC'], axis=1, inplace=True)

    print("4 result data columns:", data.columns)
    cleardata = data

    ## 数据合并（通过ID）
    data_group = cleardata.groupby('USER_ID')  # 分组

    ## 创建索引
    label = data_group[['USER_ID', 'IS_LOST']].tail(1)  # 取用户id、标记（每组的最后一个值）
    label.set_index('USER_ID', inplace=True)  # 将“USER_ID”设为索引
    print(label)


    # 3.2 去重，合并性别，年龄，终端类型。取第一次
    data_1 = data_group[['CUST_SEX', 'CERT_AGE', 'TERM_TYPE','']].first() 
    print("data 5.1\n",data_1)

    # 3.3 去重，在网时长，选择最后一次的数据
    data_2 = data_group['INNET_MONTH'].last()
    print("data 5.2\n",data_2)



    ##数据合并

    ##合约有效情况
    def cal_is_agree(x):  # x 为每个用户的三个月值
        # 如果三个月不全为1，用第三个月值减去前两个月均值；三个月的值都为1，取值为1.5。
        # 所有取值情况为-1、-0.5、0、0.5、1、1.5
        x = np.array(x)
        if x.sum() == 3:
            return 1.5
        else:
            return x[2] - x[:2].mean()

    data_3 = pd.DataFrame(data_group['IS_AGREE'].agg(cal_is_agree))#agg是一个聚合函数，聚合函数操作始终是在轴（默认是列轴，也可设置行轴）上执行，
    print("data 5.3\n",data_3)

    ##处理缺失值
    # 3.5 insert info into the label('AGREE_EXP_DATE') agree date
    date = data_group['AGREE_EXP_DATE'].last()  # 取第3个月的"合约计划到期时长"
    num_mon = (pd.to_datetime(date, format='%Y%m') - pd.to_datetime('2016-03')).dt.days/30  # 时长以“月”为单位
    data_4 = pd.DataFrame(num_mon).fillna(-1)    #用-1填充缺失值

    print("data 5.4\n",data_4)


    #*********************************************************************************************************************************************
    # 3.6 insert info into the label('CREDIT_LEVEL') level
    data_5 = pd.DataFrame(data_group['CREDIT_LEVEL'].agg('mean'))    # 信用等级
    print("data 5.5\n",data_5)


    # 3.7 缺失值-VIP等级
    data_6 = data_group['VIP_LVL'].last().fillna(0)    # 取最后一个值
    print("data 5.6\n",data_6)

    # 3.8 本月费用(取三个月的平均值)特征构建
    data_7 = pd.DataFrame(data_group['ACCT_FEE'].mean())
    print("data 5.7\n",data_7)


    # 3.9 平均每次通话时长
    # 总通话
    data_8_1 = pd.DataFrame(data_group['CALL_DURA'].sum()/data_group['CDR_NUM'].sum(),
                            columns=['Total_mean'])
    # 本地通话
    data_8_2 = pd.DataFrame(data_group['NO_ROAM_LOCAL_CALL_DURA'].sum()/data_group['NO_ROAM_LOCAL_CDR_NUM'].sum(),
                            columns=['Local_mean'])
    # 国内长途通话
    data_8_3 = pd.DataFrame(data_group['NO_ROAM_GN_LONG_CALL_DURA'].sum() / data_group['NO_ROAM_GN_LONG_CDR_NUM'].sum(),
                            columns=['GN_Long_mean'])
    # 国内漫游通话
    data_8_4 = pd.DataFrame(data_group['GN_ROAM_CALL_DURA'].sum() / data_group['GN_ROAM_CDR_NUM'].sum(),
                            columns=['GN_Roam_mean'])
    # 数据拼接
    data_8 = pd.concat([data_8_1, data_8_2, data_8_3, data_8_4], axis=1).fillna(0)

    print("data 5.8\n",data_8.head())


    # 3.10 其他变量
    # 非漫游通话次数（次）、短信发送数（条）、上网流量(MB)、本地非漫游上网流量(MB)、国内漫游上网流量(MB)、
    # 有通话天数、有主叫天数、有被叫天数  （主叫 + 被叫 ≠ 总通话）
    # 语音呼叫圈、主叫呼叫圈、被叫呼叫圈
    data_9 = data_group[['NO_ROAM_CDR_NUM', 'P2P_SMS_CNT_UP', 'TOTAL_FLUX', 'LOCAL_FLUX','GN_ROAM_FLUX',
                        'CALL_DAYS', 'CALLING_DAYS', 'CALLED_DAYS',
                        'CALL_RING','CALLING_RING', 'CALLED_RING']].agg('mean')
    print("data 5.9\n",data_9)


    #*********************************************************************************************************************************************
    # 对所有特征&标签按索引重新排序，以保证数据拼接时索引一致
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
    # 拼接所有特征&标记
    data_new = pd.concat([data_1, data_2, data_3, data_4,
            data_5, data_6, data_7, data_8, data_9, label], axis=1)
    data_new.head()

    # 4 drop nall data
    #缺失值处理

    print("6 isnull \n",data_new.isnull().sum())    # 查看缺失值
    data_new = data_new.fillna(method='ffill').fillna(method='bfill')      # 近邻值填充(向下填充+向上填充)

    data_new.to_csv('C:/Users/yizhuo/Desktop/school_resource/互联网商业智能/实训项目3.通讯运营商客户流失预测/clear_data.csv', index=True, encoding='utf-8-sig')


data = pd.read_csv('C:/Users/yizhuo/Desktop/school_resource/互联网商业智能/实训项目3.通讯运营商客户流失预测/clear_data.csv', index_col=0)
corr = data.corr()    # 皮尔逊相关系数    筛选与客户流失显著相关的特征
corr
print(data['IS_LOST'])



##############################################################################################################

## 提取特征与标记
# 以0.08作为筛选阈值
feature_index = corr['IS_LOST'].drop('IS_LOST').abs() > 0.08    # 取出与"标记"的相关系数
feature_name = feature_index.loc[feature_index].index           # 选出的重要特征名
print(feature_name)


X = data.loc[:, feature_name]    # 样本自变量
y = data.loc[:, 'IS_LOST']       # 样本目标变量
# 样本不平衡 
y.value_counts()


index_positive = y.index[y == 1]          # 正样本的索引
index_negative = np.random.choice(a=y.index[y == 0].tolist(), size=y.value_counts()[1])   # 负样本的索引，对负样本进行下采样操作

X_positive = X.loc[index_positive, :]     # 正样本自变量
X_negative = X.loc[index_negative, :]     # 负样本自变量

y_positive = y.loc[index_positive]        # 正样本标签
y_negative = y.loc[index_negative]        # 负样本标签

X = pd.concat([X_positive, X_negative], axis=0)    # 处理后的正样本
y = pd.concat([y_positive, y_negative], axis=0)    # 处理后的负样本

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # 按照y的比例，分层采样

# def Data_visualization():

##模型-随机森林
def RandomForest():
    rfc = RandomForestClassifier()    # 初始化随机森林模型

    rfc.fit(X_train, y_train)         # 模型训练

    y_pre = rfc.predict(X_test)       # 调用模型对测试样本进行预测
    print(classification_report(y_test, y_pre))    # 打印分类报告（包含了各模型性能评价指标）
    lr_acc = round(accuracy_score(y_pre,y_test)*100,2)
    print(f"logistic accuracy is: {lr_acc}%")


##模型--决策树
def DecisionTree():
    # 创建决策树模型
    dtc = DecisionTreeClassifier()

    # 训练模型
    dtc.fit(X_train,y_train)
    # 预测训练集和测试集结果
    dtc_pred = dtc.predict(X_test)

    # 计算精确度
    dtc_acc = round(accuracy_score(dtc_pred,y_test)*100,2)
    print(f"decision tree accuracy is: {dtc_acc}%")

#模型-逻辑回归
def Logistic():
    lr = LogisticRegression()

    # 训练模型
    lr.fit(X_train,y_train)

    # 预测训练集和测试集结果
    lr_pred = lr.predict(X_test)

    # 计算精确度
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

    # 定义深度学习模型
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train1.shape[1],)),
        # layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train_scaled, y_train1, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test1))


    
if __name__=='__main__':
    # data_processing()
    # Data_visualization()
    # RandomForest()    #78.25%
    # DecisionTree()  #69.83%
    depth()      #0.7905
    # Logistic()     #77.45%