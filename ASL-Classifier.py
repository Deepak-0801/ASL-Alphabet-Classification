import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# Load the dataset
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.25)

train_generator = datagen.flow_from_directory(
        'asl_alphabet_train',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = datagen.flow_from_directory(
        'asl_alphabet_train',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

# Build the model
model = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(29, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size)

# Evaluate the model
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'asl_alphabet_test',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples/test_generator.batch_size)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
