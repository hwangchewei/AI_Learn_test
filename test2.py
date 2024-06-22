import os, shutil  # 解壓縮資料夾所在的⽬錄路徑\n
original_dataset_dir = r'D:/programming_language/python/AIlearn/train' #⽤來儲存少量資料集的⽬錄位置
base_dir = r'D:/programming_language/python/AIlearn'
if not os.path.isdir(base_dir): os.mkdir(base_dir) # 如果⽬錄不存在, 才建立⽬錄 # 分拆成訓練、驗證與測試⽬錄位置
train_dir = os.path.join(base_dir,'small_train')
if not os.path.isdir(train_dir): os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'small_validation')
if not os.path.isdir(validation_dir):os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'smalltest')
if not os.path.isdir(test_dir): os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir,'cats')
if not os.path.isdir(train_cats_dir): os.mkdir(train_cats_dir) # ⽤來訓練貓圖片的⽬錄位置
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.isdir(train_dogs_dir): os.mkdir(train_dogs_dir) # ⽤來訓練狗圖片的⽬錄位置
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.isdir(validation_cats_dir): os.mkdir(validation_cats_dir) # ⽤來驗證貓圖片的⽬錄位置\n"
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.isdir(validation_dogs_dir): os.mkdir(validation_dogs_dir) # ⽤來驗證狗圖片的⽬錄位置
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.isdir(test_cats_dir): os.mkdir(test_cats_dir) # ⽤來測試貓圖片的⽬錄位置
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.isdir(test_dogs_dir): os.mkdir(test_dogs_dir) # ⽤來測試狗圖片的⽬錄位置\n"]



# fnames = ['cat.{}.jpg'.format(i) for i in range(1000*5)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst) # 複製下 500 張貓圖片到 
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000*5,1500*5)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500*5,2000*5)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000*5)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000*5,1500*5)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500*5,2000*5)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname) 
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)
    
print('訓練⽤的貓照片張數:',len(os.listdir(train_cats_dir)))
print('訓練⽤的狗照片張數:',len(os.listdir(train_dogs_dir)))
print('驗證⽤的貓照片張數:',len(os.listdir(validation_cats_dir)))
print('驗證⽤的狗照片張數:',len(os.listdir(validation_dogs_dir)))
print('測試⽤的貓照片張數:',len(os.listdir(test_cats_dir)))
print('測試⽤的狗照片張數:',len(os.listdir(test_dogs_dir)))



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
from keras.utils.np_utils import to_categorical

data = []
labels = []
imagePaths = []
dir_path = 'D:/programming_language/python/AIlearn/train'

for imagePath in glob.glob(dir_path + '/**/*' , recursive=True):
    
    if 'jpg' in imagePath:   
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (150, 150))
        data.append(image)
    
        # 读取标签
        if 'dog' in imagePath:
            label = 0
            labels.append(label)
        else:
            label = 1
            labels.append(label)
data = np.array(data,dtype=float)
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

# train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True, )
# test_datagen = ImageDataGenerator(rescale=1./255) # 請注意！驗證資料不應該擴充!!!
# train_generator = train_datagen.flow_from_directory(train_cats_dir,target_size=(150,150),batch_size=32,class_mode='binary') 

# validation_generator =test_datagen.flow_from_directory(validation_cats_dir,target_size=(150, 150),batch_size=32, class_mode='binary')
#設定訓練、測試資料的 Python 產⽣器，並將圖片像素值依 1/255 比例重新壓縮到 [0, 1]
# train_datagen = ImageDataGenerator(rescale=1./255) 
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(train_dir,target_size = (150, 150), batch_size=20,class_mode = 'binary') #因為使⽤⼆元交叉熵 binary_crossentropy 作為損失值，所以需要⼆位元標籤
# validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150), batch_size=20, class_mode='binary')


# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3),activation='relu',input_shape=(150, 150,3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3),activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128,(3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3, 3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512,activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.summary() # 查看模型摘要"

# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
# history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30, validation_data=validation_generator,validation_steps=50)


# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1,len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs,val_acc, 'b', label='Validation acc')
# plt.title('Training and validationaccuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo',label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validationloss')
# plt.title('Training and validationloss')
# plt.legend()


trainY = to_categorical(trainY,2)

testY = to_categorical(testY,2)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),activation='relu', input_shape=(150, 150,3),data_format='channels_last'))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128,(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.summary()
model.add(layers.Conv2D(128, (3, 3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.7)) # 在這裡加入 Dropout 層(丟棄 50 %)
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5)) # 在這裡加入 Dropout 層(丟棄 50 %)
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.2)) # 在這裡加入 Dropout 層(丟棄 50 %)

model.add(layers.Dense(2,activation='sigmoid'))
model.compile(optimizer=optimizers.SGD(lr=0.001),loss='categorical_crossentropy', metrics=['acc'])
historya = model.fit(x = trainX,y = trainY,epochs=100 ,validation_data=(testX,testY),batch_size= 25)

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
# test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),  batch_size=20, class_mode='binary')
# from sklearn import metrics
# test_loss,test_acc=model.evaluate_generator(test_generator,steps=50)
# print('test_acc= ',test_acc)
# z = model.predict_generator(test_generator)
# print(z)
data_test = []
imagePaths = []
dir_path = 'D:/programming_language/python/AIlearn/small_test'

for imagePath in glob.glob(dir_path + '/**/*' , recursive=True):
    if 'jpg' in imagePath:   
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (150, 150))
        data_test.append(image)
    
data_test = np.array(data_test,dtype=float)

cat = 0
dog = 0
z = model.predict(data_test)
z = z.tolist()
for x in z:
    if float(x[0])*100 > 50:
        dog = dog + 1
    else:
        cat = cat + 1
print(cat,dog)
