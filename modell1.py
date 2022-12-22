from gc import callbacks
import pandas as pd
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
#Loading the dataset
d=pd.read_csv("C:/Users/91994/Desktop/pt1CreditCardFraudDetecttion/creditcarsfrauddetectiondataset.csv")
#To check if any null values are present in dataset
is_null=d.isnull().sum().values
pos=d.value_counts(d['Class']==1)
if all(is_null)==0:#checking if all values in array are 0/zero
    #to check if the dataset is balanced or not
    '''comparison=pd.value_counts(d['Class'],sort=True)
    comparison.plot(kind='bar',rot=0)
    LABELS=['Normal','Fraud']
    plt.xticks(range(2),LABELS)
    plt.title('284315 normal and 492 fraud transactions')
    plt.xlabel('0 represents Legitimate transactions and 1 represents fraudelent trannsactions')
    plt.ylabel("Number of transactions")
    plt.xlabel("0 represents legitimate transactions and 1 repesents fraudelent transactions")
    plt.ylabel("Number of transactions")
    plt.show()
    #0 represents legitimate transactions and 1 represents fraudelent transactions
    #0- 2,84,315(transactions) and 1-492(transactions)
    #which means the dataset is highly imbalanced
    #lets look for the co-relations between the features with the help of heatmap from seaborn
    import seaborn as sns
    cr=d.corr()
    top_corr=cr.index
    plt.figure(figsize=(40,35))
    g=sns.heatmap(d[top_corr].corr(),annot=True,cmap='RdYlGn')
    plt.show()'''
    
    s=StandardScaler()
    x=s.fit_transform(d.iloc[:,1:30:1].values)
    y=d[['Class']].values
    pp=pos.values
    es=tf.keras.callbacks.EarlyStopping(monitor='val_prc',verbose=1,patience=10,mode='max',restore_best_weights=True)
    pos=pp[1]
    neg=pp[0]
    total=pos+neg
    weight_zero=(1/neg)*(total/2.0)
    weight_one=(1/pos)*(total/2.0)
    weights={0:weight_zero,1:weight_one}
    xtr,xt,ytr,yt=train_test_split(x,y,test_size=0.2)
    x_val=xtr[:len(xtr)//2]
    y_val=ytr[:len(ytr)//2]
    xtr=xtr[len(xtr)//2:]
    ytr=ytr[len(ytr)//2:]
    model=keras.models.Sequential()
    model.add(keras.layers.Dense(29 , input_dim=29,activation='relu',kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(15,activation='relu',kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(xtr,ytr,class_weight=weights,callbacks=[es],batch_size=2084,epochs=80,validation_data=(x_val,y_val))
    yp=model.predict(xt)
    yp=np.array(yp)
    ypp=[]
    for i in yp:
        if i>=0.5:
            ypp.append([1])
        else:
            ypp.append([0])
    '''model.save("model1l.h5")'''
              
    
    
    
    
    
    
    
    
