from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
import config

batch_size = 32


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    config.VAL_DIR,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)




print("### Informasi Dataset ###")
print(f"Classes found: {train_generator.class_indices}")
print(f"Number of samples: {train_generator.samples}")

x_batch, y_batch = next(train_generator)
print("\n### Batch Pertama ###")
print(f"Shape of x_batch: {x_batch.shape}")
print(f"Shape of y_batch: {y_batch.shape}")
print(f"First 5 labels: {y_batch[:5]}")




model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    

    layers.Dense(128, activation='relu'),
    

    layers.Dense(1, activation='sigmoid')
])



optimizer = Adam(learning_rate=0.001)


model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

model.save('model.h5')
print("Model saved to model.h5")
