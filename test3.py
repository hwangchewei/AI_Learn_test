import os, shutil  # 解壓縮資料夾所在的⽬錄路徑\n
# original_dataset_dir = r'D:/programming_language/python/AIlearn/train' #⽤來儲存少量資料集的⽬錄位置
# base_dir = r'D:/programming_language/python/AIlearn'
# if not os.path.isdir(base_dir): os.mkdir(base_dir) # 如果⽬錄不存在, 才建立⽬錄 # 分拆成訓練、驗證與測試⽬錄位置
# train_dir = os.path.join(base_dir,'small_train')
# if not os.path.isdir(train_dir): os.mkdir(train_dir)
# validation_dir = os.path.join(base_dir, 'small_validation')
# if not os.path.isdir(validation_dir):os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'smalltest')
# if not os.path.isdir(test_dir): os.mkdir(test_dir)
# train_cats_dir = os.path.join(train_dir,'cats')
# if not os.path.isdir(train_cats_dir): os.mkdir(train_cats_dir) # ⽤來訓練貓圖片的⽬錄位置
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# if not os.path.isdir(train_dogs_dir): os.mkdir(train_dogs_dir) # ⽤來訓練狗圖片的⽬錄位置
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# if not os.path.isdir(validation_cats_dir): os.mkdir(validation_cats_dir) # ⽤來驗證貓圖片的⽬錄位置\n"
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# if not os.path.isdir(validation_dogs_dir): os.mkdir(validation_dogs_dir) # ⽤來驗證狗圖片的⽬錄位置
# test_cats_dir = os.path.join(test_dir, 'cats')
# if not os.path.isdir(test_cats_dir): os.mkdir(test_cats_dir) # ⽤來測試貓圖片的⽬錄位置
# test_dogs_dir = os.path.join(test_dir, 'dogs')
# if not os.path.isdir(test_dogs_dir): os.mkdir(test_dogs_dir) # ⽤來測試狗圖片的⽬錄位置\n"]

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from tensorflow import keras
from keras import optimizers
from tensorflow import optimizers
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import glob
from tensorflow.keras.utils import to_categorical

data = []
labels = []
imagePaths = []
dir_path = 'D:/programming_language/python/AIlearn/train'
count,c_dog,c_cat = 0,0,0
for imagePath in glob.glob(dir_path + '/**/*' , recursive=True):
    # print(imagePath)
    # break
    if '.jpg' in imagePath:   
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (150, 150))
        
    
        # 读取标签
        if 'dog' in imagePath:
            if c_dog > 6003:
                continue
            c_dog+=1
            data.append(image)
            label = 0
            labels.append(label)
        elif 'cat' in imagePath:
            if c_cat > 6003:
                continue
            c_cat+=1
            data.append(image)
            label = 1
            labels.append(label)
    
    count+=1
    # print(labels)
    # break
    if count >= 12000:
        break
# print(labels)
data = np.array(data,dtype=float)
labels = np.array(labels)

print(data.shape)

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

trainY = to_categorical(trainY,2)

testY = to_categorical(testY,2)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),activation='relu', input_shape=(150, 150,3),data_format='channels_last'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128,(3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3, 3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())


model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5)) # 在這裡加入 Dropout 層(丟棄 50 %)
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5)) # 在這裡加入 Dropout 層(丟棄 50 %)
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.2)) # 在這裡加入 Dropout 層(丟棄 50 %)

model.add(layers.Dense(2,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
historya = model.fit(x = trainX,y = trainY,epochs=30 ,validation_data=(testX,testY),batch_size= 100)

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

data_test = []
imagePaths = []
a = []
dir_path = 'D:/programming_language/python/AIlearn/small_test'

for imagePath in glob.glob(dir_path + '/**/*' , recursive=True):
    if 'jpg' in imagePath:   
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (150, 150))
        data_test.append(image)
        a.append(imagePath)
data_test = np.array(data_test,dtype=float)

cat = 0
dog = 0
z = model.predict(data_test)
print(z)

for x in z:
    if float(x[0])*100 > 50:
        dog = dog + 1
    else:
        cat = cat + 1
print(cat,dog)
# print(a[260])
# print(model.predict(data_test[2603:2604]))
# plt.figure()
# plt.bar(range(2),model.predict(data_test[2603:2604]).flatten())
# plt.xticks(range(3))
# plt.show()


