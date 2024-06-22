from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from tensorflow import keras
import tensorflow as tf
from keras import optimizers
from tensorflow import optimizers
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import glob
from tensorflow.keras.utils import to_categorical
import random
from tqdm import tqdm

data = []
labels = []
imagePaths = []
dir_path = 'E:/PubChem'
maybelist = [10548,5965,5966,4991,7550,202190,134575,98821,3152,5741,3655]
# maybelist = [1933,1935,3655,22493,179849,192272]
maybelist.sort()
count = 0
x = 0
randomlist = random.sample(range(800000),500)
for y in randomlist:
    imagePath = dir_path+'\\'+str(y)+'.png'
    if imagePath == (dir_path+'\\'+str(maybelist[x])+'.png'):
        x+=1
        continue
    else:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (150, 150))
        data.append(image)
        labels.append(0)
        
data = np.array(data,dtype=float)
label = np.array(labels)        
# sample_num = int(0.5 * len(labels)) # 假設取50%的資料
# sample_list = [i for i in range(len(data))] # [0, 1, 2, 3]
# sample_list = random.sample(sample_list, sample_num) # [1, 2]
# data = data[sample_list,:]
# label = labels[sample_list] # array([2, 3])
# print(data.shape,labels.shape) 
 
datatrue = []
labelstrue = []                          
dir_path = 'E:cid'

for imagePath in glob.glob(dir_path + '/*' , recursive=True):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (150, 150))
    datatrue.append(image)
    labelstrue.append(1)
    datatrue.append(tf.image.flip_left_right(image))
    labelstrue.append(1)
    datatrue.append(tf.image.flip_up_down(image))
    labelstrue.append(1)
    datatrue.append(tf.image.transpose(image))
    labelstrue.append(1)
    # a = tf.image.resize_with_crop_or_pad(image,random.randint(1,250),random.randint(1,250))
    # datatrue.append(tf.image.resize(a, (150, 150)))
    # labelstrue.append(1)
    
    



datatrue = np.array(datatrue,dtype=float)
labelstrue = np.array(labelstrue)  
print(datatrue.shape,labelstrue.shape)
data = np.append(data,datatrue,axis=0)
label = np.append(label,labelstrue,axis=0)
print(len(data))
(trainX, testX, trainY, testY) = train_test_split(data,label, test_size=0.1, random_state=20)

trainY = to_categorical(trainY,2)

testY = to_categorical(testY,2)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),activation='relu', input_shape=(150, 150,3),data_format='channels_last'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128,(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# model.add(layers.Conv2D(256, (4, 4),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())


model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5)) # 在這裡加入 Dropout 層(丟棄 50 %)
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5)) # 在這裡加入 Dropout 層(丟棄 50 %)
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.2)) # 在這裡加入 Dropout 層(丟棄 50 %)

model.add(layers.Dense(2,activation='sigmoid'))
model.summary()
model.compile(optimizer='adagrad',loss='binary_crossentropy', metrics=['acc'])
historya = model.fit(x = trainX,y = trainY,epochs=30 ,validation_split = 0.2,batch_size= 20)

# # historya = model.fit_generator(train_generator,steps_per_epoch=10,epochs=3 ,validation_data=validation_generator,validation_steps=50)
acca = historya.history['acc']
val_acca = historya.history['val_acc']
lossa = historya.history['loss']
val_lossa = historya.history['val_loss']
epochsa = range(1,len(acca) + 1)
plt.plot(epochsa, acca, 'g', label='Training acc')
plt.plot(epochsa,val_acca, 'b', label='Validation acc')
plt.title('Training and validationaccuracy')
plt.legend()
plt.figure()
plt.plot(epochsa, lossa, 'g',label='Training loss')
plt.plot(epochsa, val_lossa, 'b', label='Validation loss')
plt.title('Training and validationloss')
plt.legend()
plt.show()

# print(model.evaluate(testX,testY))
dir_path = 'E:/PubChem'

path = []
count = 0
truepredict = []
x = 0
progress = tqdm(total=2000)
for y in range(1,2000):
    if y == maybelist[x]:
        data_test = []
        imagePath = dir_path+'\\'+str(maybelist[x])+'.png'
        x+=1
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (150, 150))
        data_test.append(image)
        data_test = np.array(data_test,dtype=float)
        z = model.predict(data_test[0:1])
        print(z[0])
    imagePath = dir_path+'\\'+str(y)+'.png'
    count += 1
    data_test = []
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (150, 150))
    data_test.append(image)
    data_test = np.array(data_test,dtype=float)
    z = model.predict(data_test[0:1],verbose=0)
    if imagePath == (dir_path+'\\'+str(maybelist[x])+'.png'):
        x+=1
        truepredict.append(y)
        truepredict.append(z)
    elif 7 < z[0][1]*10:       
        path.append(y)

    progress.update(1)
    

print(path)
print(truepredict)
image = cv2.imread('E:/PubChem/250817.png')
image = cv2.resize(image, (150, 150))
data_test=[image]
data_test = np.array(data_test,dtype=float)
z = model.predict(data_test[0:1],verbose=0)
print('E:/PubChem/250817.png')
print(z[0])
