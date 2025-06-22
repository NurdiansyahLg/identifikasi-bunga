import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import numpy as np
import json

# Data augmentation + preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    'Dataset/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'Dataset/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[early_stop])

# === Evaluasi Model pada Data Validasi ===
val_generator.reset()
y_pred = model.predict(val_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
target_names = list(val_generator.class_indices.keys())

report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True)

# Cetak metrik utama
print("=== Evaluasi Model ===")
print("Akurasi:", round(report['accuracy'] * 100, 2))
print("Precision:", round(report['macro avg']['precision'] * 100, 2))
print("Recall:", round(report['macro avg']['recall'] * 100, 2))
print("F1-Score:", round(report['macro avg']['f1-score'] * 100, 2))

# Save model
model.save('model_bunga.h5')

# Save label mapping
with open('label_map.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
