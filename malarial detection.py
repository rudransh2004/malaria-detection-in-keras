from PIL import Image
import numpy as np
import os
import cv2
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout




data = []
labels = []

Parasitized=os.listdir("cell_images/cell_images/Parasitized/")

for a in os.listdir("cell_images/cell_images/Parasitized/"):
    try:
        image = cv2.imread("cell_images/cell_images/Parasitized/"+a)
        image_from_array = Image.fromarray(image,'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(0)
        
    except AttributeError:
        print("")
        

uninfected = os.listdir("cell_images/cell_images/Parasitized/")

for a in os.listdir("cell_images/cell_images/Parasitized/"):
    try:
        image = cv2.imread("cell_images/cell_images/Parasitized/"+a)
        image_from_array = Image.fromarray(image,'RGB')
        size_image = image_from_array.resize((50,50))
        data.append(np.array(size_image))
        labels.append(0)
        
    except AttributeError:
        print("")
        
Cells=np.array(data)
labels=np.array(labels)


np.save("Cells",Cells)
np.save("labels",labels)

Cells=np.load("Cells.npy")
labels=np.load("labels.npy")


s=np.arange(Cells.shape[0])
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

num_classes=len(np.unique(labels))
len_data=len(Cells)

(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)

(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1 ,activation="softmax"))#2 represent output layer neurons 
model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=50,epochs=5,verbose=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("malaria_detector.h5")
print("Saved model to disk")




        
        
        
        
        

