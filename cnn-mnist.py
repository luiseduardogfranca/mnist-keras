from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import mnist 
import matplotlib.pyplot as plt
import numpy as np 

'''
    TODO    
    'input shape'
    'num_classes'

    Label	Description
    0	T-shirt/top
    1	Trouser
    2	Pullover
    3	Dress
    4	Coat
    5	Sandal
    6	Shirt
    7	Sneaker
    8	Bag
    9	Ankle boot
'''

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = [] 
    def on_epoch_train(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def plotImage(image, gray=True, title='Number'):
    if(gray):
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    plt.title(title)
    plt.show()

def getScore(model, x_test, y_test):
    return model.evaluate(x_test, y_test, verbose=0)

# get dataset mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train[:], axis=3)
x_test = np.expand_dims(x_test[:], axis=3)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# expand the dimension for get input shape
input_shape = x_train[0].shape

# get the number of class 
class_mnist = [0,1,2,3,4,5,6,7,8,9]
num_classes = len(class_mnist)

batch_size = 128
epochs = 10

# instance of history for callback
history = AccuracyHistory()

# define model 
model = Sequential()

# add first layer 2D with 
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1,1), activation='relu', input_shape=input_shape))

# add layer pooling 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# add other conv2D
model.add(Conv2D(64, (5,5), activation="relu"))

# add layer pooling 
model.add(MaxPooling2D(pool_size=(2,2)))

# create fully connected layer 
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation="softmax"))

# set loss function(cross entropy) and optimizer(Adam)
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.01), metrics=['accuracy'])

# train model with dataset of train 
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test), callbacks=[history], shuffle=False)

# testing the power of prediction
score = getScore(model, x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

index = 5000
images = np.expand_dims(x_train[index], axis=0)

result = model.predict_classes(images)
print("Class correct:", [i for i in range(len(y_train[index])) if y_train[index,i] == 1])
print("Class predict:", result)

plotImage(np.squeeze(x_train[index]))
