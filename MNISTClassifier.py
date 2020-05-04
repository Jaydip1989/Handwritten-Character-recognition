import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import cv2
batch_size = 128
epochs = 5

# loading the dataset
(X_train,y_train),(X_test,y_test) = mnist.load_data()

img_rows = X_train[0].shape[0]
img_cols = X_test[0].shape[0]

X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)


input_shape = (img_rows,img_cols,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizing the data
X_train /= 255
X_test /= 255

print("X_train shape", X_train.shape)
print('Train samples', X_train.shape[0])
print('Test samples', X_test.shape[0])

# We one hot encode the outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print('Number of classes:' + str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = X_train.shape[1] * X_train.shape[2]

## Create Model
model = Sequential()
model.add(Conv2D(32,kernel_size = (3,3),activation = 'relu',
                 input_shape = input_shape))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation = "softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

print(model.summary())

history = model.fit(X_train,y_train,batch_size = batch_size ,epochs = epochs,
                    verbose = 1,
                    validation_data = (X_test,y_test))
score = model.evaluate(X_test,y_test,verbose = 0)
print("Test Accuracy",score[1]*100)
print("Test Loss",score[0]*100)

model.save('mnist_simple_cnn.h5')