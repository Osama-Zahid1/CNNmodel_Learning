import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")
import matplotlib
print(f"Matplotlib version: {matplotlib.__version__}")

dataset_zip = 'ChineseHerbs.zip' 
extracted_dir = r'#folder where u upload the zip file'  

if os.path.exists(dataset_zip):
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)
    print("Dataset extracted successfully.")
else:
    print("Dataset zip file not found. Please upload it.")

print("\nChecking extracted dataset structure...")
for root, dirs, files in os.walk(extracted_dir):
    print(f"Directory: {root}, Number of Files: {len(files)}")

base_dir = 'ChineseHerbs/ChineseHerbs/'  

if not os.path.exists(base_dir):
    raise FileNotFoundError(f"Base directory '{base_dir}' not found. Please check your dataset structure.")
    
def display_sample_images(base_dir):
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    fig, axes = plt.subplots(len(categories), 2, figsize=(10, len(categories) * 3))
    for i, category in enumerate(categories):
        category_path = os.path.join(base_dir, category)
        sample_images = os.listdir(category_path)[:2]  
        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(category_path, img_name)
            img = plt.imread(img_path)
            axes[i, j].imshow(img)
            axes[i, j].set_title(category)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

display_sample_images(base_dir)


# Define image dimensions
img_height, img_width = 224, 224

# Data generators
data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Training data generator
train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Define the baseline model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_generator, validation_data=val_generator, epochs=10)

print("Baseline model training completed.")

# Define the Improved CNN Model
enhanced_model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(train_generator.num_classes, activation='softmax')  
])

enhanced_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

enhanced_history = enhanced_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)

results = enhanced_model.evaluate(val_generator)
print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")

# Evaluate Baseline Model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred_baseline = np.argmax(model.predict(val_generator), axis=1)
baseline_cm = confusion_matrix(val_generator.classes, y_pred_baseline)
baseline_report = classification_report(val_generator.classes, y_pred_baseline, target_names=list(train_generator.class_indices.keys()))

print("Baseline Model Classification Report:\n", baseline_report)
print("Baseline Model Confusion Matrix:\n", baseline_cm)

# Evaluate Enhanced Model
y_pred_enhanced = np.argmax(enhanced_model.predict(val_generator), axis=1)
enhanced_cm = confusion_matrix(val_generator.classes, y_pred_enhanced)
enhanced_report = classification_report(val_generator.classes, y_pred_enhanced, target_names=list(train_generator.class_indices.keys()))

print("Enhanced Model Classification Report:\n", enhanced_report)
print("Enhanced Model Confusion Matrix:\n", enhanced_cm)

