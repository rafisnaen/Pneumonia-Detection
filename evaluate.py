
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
from sklearn.metrics import confusion_matrix, classification_report

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = val_test_datagen.flow_from_directory(
    config.TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

model = tf.keras.models.load_model('model.h5')
print("Model loaded from model.h5")

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_acc:.2f}")

y_pred_prob = model.predict(test_generator, batch_size=32)

y_pred = (y_pred_prob > 0.5).astype(int)

y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
