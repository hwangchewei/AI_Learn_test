import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
import matplotlib.pyplot as pt

#預測鐵達尼號


#導入檔案，並刪除與生存率不相關的數據(預處理)
titanic = pd.read_csv('D:/programming_language/python/train.csv')
a = pd.get_dummies(titanic['Embarked'])   # 將Embarked的數據轉為DataFrame的形式
titanic = titanic.drop(['Name','Cabin','Ticket','PassengerId','Fare','Embarked'],axis=1)
age_mean = titanic['Age'].mean()
titanic['Age'] = titanic['Age'].fillna(age_mean)  # 填補空格
titanic['Sex'] = titanic['Sex'].map({'male':1,'female':2}).astype(int)
titanic = titanic.join(a)
print(titanic)
print(titanic.shape)

#獲取資料的值，並拆分為訓練及預測兩類(8:2)
titanic = titanic.values
titanic_pridic = titanic[713:]
titanic = titanic[:713]
print(titanic)
print(titanic.shape)
print(titanic.flatten())
print(titanic.flatten().shape)
#將訓練資料拆為訓練集與測試集
x = titanic[:,1:]
y = titanic[:,0]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2) #(8:2)拆分x,y
print(y_test.shape)

#模型訓練
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',kernel_initializer='uniform',input_dim = 8))
model.add(layers.Dense(32,activation='relu',kernel_initializer='uniform'))
model.add(layers.Dense(32,activation='relu',kernel_initializer='uniform'))
model.add(layers.Dense(1,activation='sigmoid',kernel_initializer='uniform'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

a = model.fit(x = x_train.astype(int),y = y_train.astype(int),epochs=250,batch_size=10,validation_split=0.1,verbose=2)

#繪圖
dict_a = a.history
loss_value = dict_a['loss']
val_loss_value = dict_a['val_loss']
epoch = range(1,len(loss_value)+1)
pt.subplot(2,2,1)
pt.plot(epoch,loss_value,'b',label='train_loss')
pt.plot(epoch,val_loss_value,'g',label='val_loss')
pt.xlabel('epoch')
pt.ylabel('loss')
pt.title("plot 1")
acc = dict_a['accuracy']
val_acc = dict_a['val_accuracy']
pt.subplot(2,2,2)
pt.plot(epoch,acc,'b',label='train_acc')
pt.plot(epoch,val_acc,'g',label='val_acc')
pt.xlabel('epoch')
pt.ylabel('acc')
pt.title("plot 2")
pt.show()

#預測
titanic_pridic_racc = titanic_pridic[:,0] #資料給的  此人是否生存
titanic_pridic_data = titanic_pridic[:,1:]
titanic_pridic_acc = [] #預測生存率
z = model.predict(titanic_pridic_data.astype(np.float32))
count_acc = 0
for a in range(len(z)):
    if z[a] > 0.5: #若此人生存率大於一半，則表示存活
        z[a] = 1
    else:
        z[a] = 0
    if z[a] == titanic_pridic_racc[a]: #比較資料集預測結果
        count_acc += 1
print(count_acc/len(z)) #判斷準確率

