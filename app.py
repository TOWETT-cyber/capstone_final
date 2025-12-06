# =============================================
# Complete Image Classification with TensorFlow
# Dataset: Fashion-MNIST (built-in)
# Model: CNN built from scratch
# Features: Training callbacks, metrics, visualizations, model save/load
# =============================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# ----------------------------
# 1. Verify TensorFlow version
# ----------------------------
print("TensorFlow version:", tf.__version__)

# ----------------------------
# 2. Load and Preprocess Data
# ----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension (grayscale => (28, 28, 1))
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Class names for visualization
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Create validation split (80% train, 20% val)
val_split = int(0.2 * len(x_train))
x_val = x_train[:val_split]
y_val = y_train[:val_split]
x_train = x_train[val_split:]
y_train = y_train[val_split:]

print(f"Training samples: {x_train.shape[0]}")
print(f"Validation samples: {x_val.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# ----------------------------
# 3. Build CNN Model from Scratch
# ----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='linear')  # Output logits
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# 4. Set Up Callbacks
# ----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# ----------------------------
# 5. Train the Model
# ----------------------------
print("\nStarting training...\n")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ----------------------------
# 6. Evaluate on Test Set
# ----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Generate predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# ----------------------------
# 7. Advanced Evaluation Metrics
# ----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ----------------------------
# 8. Plot Training History
# ----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ----------------------------
# 9. Save Final Model (if not already)
# ----------------------------
model.save('final_model.keras')
print("\nModel saved as 'final_model.keras'")

# ----------------------------
# 10. Load and Verify Model
# ----------------------------
loaded_model = tf.keras.models.load_model('final_model.keras')
loaded_test_loss, loaded_test_acc = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f"Verified loaded model test accuracy: {loaded_test_acc:.4f}")

# Optional: Clean up (uncomment if desired)
# os.remove('best_model.keras')
# os.remove('final_model.keras')
