import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
import warnings
from keras.datasets import mnist

warnings.filterwarnings('ignore')

# Veri Setimizi keras'tan yukleyelim
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# rastgele goruntuler getirelim
n, k, figsize = 5, 5, (5, 5)
fig, ax = plt.subplots(k, n, figsize=figsize)
for i in range(n):
    for j in range(k):
        ax[i, j].imshow(X_train[np.random.randint(X_train.shape[0])], cmap='gray')
        ax[i, j].axis('off')

plt.show()

y_train = to_categorical(y_train) # OneHotEncoding
y_test = to_categorical(y_test) # OneHotEncoding
X_train = X_train.astype('float64') # type convert
X_test = X_test.astype('float64') # type convert

number_of_class = y_train.shape[1] # 1. indexteki class'larimizin sayisi 0'da ise resimlerin sayisi

# CNN
model = Sequential()

model.add(Conv2D(input_shape=(28, 28, 1), filters=16, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=256))
model.add(Activation('relu'))
model.add((Dropout(0.2)))

model.add(Dense(units=number_of_class))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=20,
                    batch_size=200)

model.save_weights('cnn_mnist_model.h5')

print(history.history.keys())

plt.plot(history.history['loss'], label='Loss Value')
plt.plot(history.history['val_loss'], label='Validation Loss Value')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Accuracy Value')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy Value')
plt.legend()
plt.show()

# Save History
import json
with open('cnn_mnist_history.json', mode='w', encoding='utf-8') as f:
    h = json.dump(history.history, f)

# Load History
import codecs
with codecs.open('cnn_mnist_history.json', mode='r', encoding='utf-8') as f:
    h = json.loads(f.read())

# Yuklenen History'i gorelim
plt.figure()
plt.plot(h['loss'], label='Loss')
plt.plot(h['val_loss'], label='Val Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(h['accuracy'], label='Acc')
plt.plot(h['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

