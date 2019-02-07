from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.neighbors import NearestNeighbors
from keras.datasets import cifar10
import numpy as np
from keras import backend as K

################################################################

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

################################################################
x = np.array([image.flatten() for image in x_train])
print(x[0])

# import sys
# sys.exit(0)

model = Sequential()

# hiddens
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(48, activation='relu'))

# output
model.add(Dense(img_rows * img_cols  * 3, activation='linear'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# Fit the model
model.fit(x_train, x, epochs=3, batch_size=10)

# remove last layer
model.pop()

# prepare map of points
points_values = model.predict(x_train)

print(points_values[0])
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

knn = NearestNeighbors()
knn.fit(points_values)
for image in [120, 300, 700, 1000]:
    fitting = knn.kneighbors([points_values[image]], n_neighbors=5, return_distance=False)[0]
    for fit in fitting:
        print("ID: {}, class: {}".format(fit, y_train[fit]))


# ID: 120, class: [2]
# ID: 827, class: [2]
# ID: 7509, class: [0]
# ID: 30612, class: [2]
# ID: 49719, class: [4]
# ID: 300, class: [2]
# ID: 31691, class: [2]
# ID: 2573, class: [4]
# ID: 18888, class: [2]
# ID: 40966, class: [2]
# ID: 700, class: [0]
# ID: 35599, class: [0]
# ID: 33571, class: [8]
# ID: 36547, class: [8]
# ID: 48158, class: [8]
# ID: 1000, class: [9]
# ID: 20544, class: [7]
# ID: 24979, class: [1]
# ID: 40359, class: [0]
# ID: 17782, class: [7]
