# 🖼️ Simple Image Classification with TensorFlow  
*A beginner-friendly, end-to-end machine learning project using Fashion-MNIST*

---

## 📌 Overview

This project demonstrates how to build, train, evaluate, and save a **convolutional neural network (CNN)** for image classification using **TensorFlow** and the **Fashion-MNIST dataset**—a more challenging drop-in replacement for the classic MNIST digit dataset.

It’s designed as a **complete educational template** that covers the entire machine learning workflow:
- Data loading & preprocessing  
- Model definition from scratch  
- Robust training with callbacks  
- Comprehensive evaluation (metrics + visualizations)  
- Model persistence (save/load)

No prior TensorFlow experience is required—just basic Python knowledge!

---

## 🎯 Target Audience

- **Students** learning deep learning or computer vision  
- **Developers** exploring AI for the first time  
- **Educators** needing a clean, reproducible demo  
- **Hobbyists** building personal AI projects  

> ✅ Perfect for labs, tutorials, hackathons, or self-study.

---

## 🔧 System Requirements

| Component        | Requirement |
|------------------|-------------|
| **Operating System** | Windows 10/11, macOS (12+), or Linux (Ubuntu 20.04+) |
| **RAM**          | Minimum 4 GB (8 GB recommended) |
| **Python**       | 3.9 – 3.13 ([Download Python](https://www.python.org/downloads/)) |
| **Disk Space**   | ~500 MB free |

> 💡 This project uses the built-in **Fashion-MNIST** dataset—no internet needed after initial setup.

---

## 📦 Installation & Setup

1. **Install Python** (if not already installed):  
   Go to [python.org/downloads](https://www.python.org/downloads/) and download the latest stable version (3.13+ recommended).

2. **Open a terminal or command prompt** and run:

   ```bash
   # Upgrade pip (recommended)
   python -m pip install --upgrade pip

   # Install required packages
   pip install tensorflow numpy matplotlib seaborn scikit-learn
   ```

3. **Verify TensorFlow installation**:

   ```python
   import tensorflow as tf
   print("TensorFlow version:", tf.__version__)
   ```

   ✅ Expected output: `TensorFlow version: 2.x.x` (e.g., `2.16.1`)

---

## ▶️ How to Run

1. Save the [complete code](#-complete-code) (below) as `image_classifier.py`.
2. In your terminal, run:

   ```bash
   python image_classifier.py
   ```

3. The script will:
   - Download and preprocess Fashion-MNIST
   - Train a CNN for up to 30 epochs (with early stopping)
   - Print test accuracy and classification report
   - Save two plots:  
     - `training_history.png`  
     - `confusion_matrix.png`
   - Save the trained model as `final_model.keras`  
   - Load and verify the saved model

> 🕒 Training takes **2–5 minutes** on a modern CPU.

---

## 📁 Output Files

After running, you’ll find these files in your project directory:

| File | Description |
|------|-------------|
| `final_model.keras` | Full trained model (architecture + weights + optimizer state) |
| `best_model.keras` | Best-performing model during training (automatically saved) |
| `training_history.png` | Line plots of training/validation accuracy and loss |
| `confusion_matrix.png` | Heatmap showing per-class prediction performance |

> 🔄 You can delete `best_model.keras` after successful run—it’s only used internally.

---

## 🧠 Key Concepts Demonstrated

| Concept | Why It Matters |
|-------|----------------|
| **CNN Architecture** | Uses convolutional layers to detect spatial patterns (edges, textures, shapes) |
| **Data Normalization** | Scales pixel values to `[0, 1]` for faster, stable training |
| **Train/Val/Test Split** | Ensures unbiased evaluation and prevents overfitting |
| **Callbacks** | `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` automate robust training |
| **Sparse Categorical Crossentropy** | Efficient loss for integer labels (no one-hot encoding needed) |
| **Confusion Matrix** | Reveals which classes the model confuses (e.g., “Shirt” vs “T-shirt”) |

---

## 🧪 Sample Output (Console)

```
TensorFlow version: 2.16.1
Training samples: 48000
Validation samples: 12000
Test samples: 10000

Model: "sequential"
...
Starting training...

Epoch 1/30
375/375 [==============================] - 8s 20ms/step - ...
Epoch 5/30
375/375 [==============================] - 7s 19ms/step - ...

Test Accuracy: 0.9124

Classification Report:
              precision    recall  f1-score   support
 T-shirt/top       0.88      0.85      0.86      1000
     Trouser       0.98      0.97      0.98      1000
    Pullover       0.85      0.88      0.86      1000
       Dress       0.91      0.92      0.92      1000
        Coat       0.86      0.85      0.86      1000
      Sandal       0.95      0.92      0.94      1000
       Shirt       0.78      0.73      0.75      1000
     Sneaker       0.94      0.97      0.96      1000
         Bag       0.97      0.99      0.98      1000
  Ankle boot       0.94      0.96      0.95      1000

Model saved as 'final_model.keras'
Verified loaded model test accuracy: 0.9124
```

---

## 🛠️ Customization Guide

Want to adapt this for your own project?

| Task | How to Modify |
|------|---------------|
| **Use MNIST instead** | Replace `fashion_mnist` with `mnist` in the data loading step |
| **Add more layers** | Insert additional `Conv2D` or `Dense` layers in the model |
| **Change image size** | Update `input_shape` and preprocessing if using custom data |
| **Train longer** | Increase `epochs` in `model.fit()` |
| **Use your own images** | Replace data loading with `tf.keras.utils.image_dataset_from_directory()` |

> 💡 Tip: For color images (e.g., CIFAR-10), change input shape to `(32, 32, 3)` and adjust normalization.

---

## ❓ Common Issues & Fixes

| Issue | Solution |
|------|--------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install tensorflow` in the correct Python environment |
| Training is slow | Ensure you’re not using a virtual environment with restricted resources; consider GPU if available |
| Low accuracy (<80%) | The model may be underfitting—try adding layers, increasing filters, or training longer |
| Plots don’t appear | Add `plt.show()` or run in Jupyter Notebook; some IDEs suppress GUI output |
| “Best model” not saved | Check write permissions in your project directory |

---

## 📚 References & Resources

- [TensorFlow Official Site](https://www.tensorflow.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Python Downloads](https://www.python.org/downloads/)
- [Keras Documentation](https://keras.io/)

---

## 📄 License

This project is open-source and free to use for **educational, personal, and commercial purposes** under the **Apache 2.0 License** (inherited from TensorFlow).

---

> ✨ **Happy coding!** This project is your launchpad into the world of computer vision. Fork it, break it, improve it—and build something amazing.
