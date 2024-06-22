from cProfile import label
from statistics import mode
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
import matplotlib.pyplot as pt
import numpy as np

def c(seqs,dim = 10000):
    results = np.zeros((len(seqs),dim))
    for i , seq in enumerate(seqs):
        results[i,seq] = 1.0
    return results
(train_dataa,train_labelsa) , (train_datab,train_labelsb)= imdb.load_data(num_words=10000)
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))
model.add(layers.Flatten())
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
x_train = c(train_dataa)
y_train = c(train_datab)
x_val = x_train[:10000]
y_val = y_train[:10000]
x_test = x_train[10000:]
y_test = y_train[10000:]
print(x_test.shape)
print(y_test.shape)
a = model.fit(x_test,y_test,epochs=4,batch_size=512,validation_data=(x_val,y_val))

dict_a = a.history
print(dict_a)
# loss_value = dict_a['loss']
# val_loss_value = dict_a['val_loss']
# epoch = range(1,len(loss_value)+1)
# pt.subplot(1,2,1)
# pt.plot(epoch,loss_value,'bo',label='train_loss')
# pt.plot(epoch,val_loss_value,'b',label='val_loss')
# pt.xlabel('epoch')
# pt.ylabel('loss')
# pt.title("plot 1")
# acc = dict_a['acc']
# val_acc = dict_a['val_acc']
# pt.subplot(1,2,2)
# pt.plot(epoch,acc,'bo',label='train_acc')
# pt.plot(epoch,val_acc,'b',label='val_acc')
# pt.xlabel('epoch')
# pt.ylabel('acc')
# pt.title("plot 2")
# pt.show()


