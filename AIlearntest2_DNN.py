from pickletools import float8
from matplotlib.cbook import flatten
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
import matplotlib.pyplot as pt

titanic = pd.read_csv('D:/programming_language/python/train.csv')
titanic.head()
# titanic_test = pd.read_csv('D:/programming_language/python/test.csv')
titanic = titanic.drop(['Name','Cabin','Ticket','PassengerId','Fare'],axis=1)
# titanic_test = titanic_test.drop(['Name','Cabin','Ticket','PassengerId','Fare'],axis=1)
age_mean = titanic['Age'].mean()
titanic['Age'] = titanic['Age'].fillna(age_mean)
titanic['Sex'] = titanic['Sex'].map({'male':1,'female':2}).astype(int)
# titanic_test['Age'] = titanic_test['Age'].fillna(age_mean)
# titanic_test['Sex'] = titanic_test['Sex'].map({'male':1,'female':2}).astype(int)
a = pd.get_dummies(titanic['Embarked'])
a = a['S'].map(str)+a['Q'].map(str)+a['C'].map(str)
# b = pd.get_dummies(titanic_test['Embarked'])
# b = b['S'].map(str)+b['Q'].map(str)+b['C'].map(str)



titanic['Embarked'] = a
# titanic_test['Embarked'] = b


# for x in range(len(titanic['Embarked'])):
#     c = str(a[x].tolist()[0])+str(a[x].tolist()[1])+str(a[x].tolist()[2])
#     titanic['Embarked'][x] = c

    
print(titanic)
titanic = titanic.values
print(titanic)
# for x in range(len(titanic_test['Embarked'])):
#     d = str(b[x].tolist()[0])+str(b[x].tolist()[1])+str(b[x].tolist()[2])
#     titanic_test['Embarked'][x] = d

# print(titanic['Embarked'])    

# titanic_test = titanic_test.values

titanic_test = titanic[713:]
titanic = titanic[:713]
x = titanic[:,1:]
y = titanic[:,0]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# x_train,x_test,y_train,y_test = x[:713],x[713:],y[:713],y[713:]
print(y_train.shape)
print(x_train.shape)
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',kernel_initializer='uniform',input_dim = 6))
model.add(layers.Dense(32,activation='relu',kernel_initializer='uniform'))
model.add(layers.Dense(32,activation='relu',kernel_initializer='uniform'))
model.add(layers.Dense(1,activation='sigmoid',kernel_initializer='uniform'))
# model.add(layers.Flatten())
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

# a = model.fit(x = x_train.astype(int),y = y_train.astype(int),epochs=250,batch_size=10,validation_split=0.1,verbose=2)

# dict_a = a.history
# loss_value = dict_a['loss']
# val_loss_value = dict_a['val_loss']
# epoch = range(1,len(loss_value)+1)
# pt.subplot(2,2,1)
# pt.plot(epoch,loss_value,'b',label='train_loss')
# pt.plot(epoch,val_loss_value,'g',label='val_loss')
# pt.xlabel('epoch')
# pt.ylabel('loss')
# pt.title("plot 1")
# acc = dict_a['accuracy']
# val_acc = dict_a['val_accuracy']
# pt.subplot(2,2,2)
# pt.plot(epoch,acc,'b',label='train_acc')
# pt.plot(epoch,val_acc,'g',label='val_acc')
# pt.xlabel('epoch')
# pt.ylabel('acc')
# pt.title("plot 2")
# pt.show()
# titanic_test_racc = titanic_test[:,0]
# titanic_test_data = titanic_test[:,1:]
# titanic_test_acc = []
# print(titanic_test)
# z = model.predict(titanic_test_data.astype(np.float32))
# count_acc = 0
# for a in range(len(z)):
#     if z[a] > 0.5:
#         z[a] = 1
#     else:
#         z[a] = 0
#     if z[a] == titanic_test_racc[a]:
#         count_acc += 1
# print(count_acc/len(z))