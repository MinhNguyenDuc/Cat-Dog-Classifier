import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

#Convolution
model.add(Conv2D(32, (3,3), input_shape= (50,50,3), activation='relu'))

#Pooling
model.add(MaxPooling2D(pool_size=(2,2)))

#Second convolution
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
model.add(Flatten())

#FCL
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (50, 50),
                                            batch_size = 32,
                                            class_mode = 'binary')


model.fit_generator(training_set,
             samples_per_epoch = 8000,
             nb_epoch = 25,
             validation_data = test_set,
             nb_val_samples = 2000)


model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./model.txt")
print("saved model..! ready to go.")

model.save('cat_dog.')




import cv2
from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

img_cat = cv2.imread("cat.jpg")

img_dog = cv2.imread("./dataset/training_set/dogs/dog.327.jpg")


#Cat
img_cat = cv2.resize(img_cat, (50,50))
print(img_cat.shape)
img_cat = img_cat.reshape(1, 50, 50, 3)

#Dog
img_dog = cv2.resize(img_dog, (50,50))
print(img_dog.shape)
img_dog = img_dog.reshape(1, 50, 50, 3)

print(loaded_model.predict_classes(img_cat))
print(loaded_model.predict(img_dog))


def preprocess_img(img):
    img = cv2.resize(img, (50, 50))
    img = img.reshape(1, 50, 50, 3)
    print(loaded_model.predict_classes(img))


a = [[0]]
print(a[0][0])









