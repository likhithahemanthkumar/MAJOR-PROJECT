import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "ai_engine/dataset/organized_stages/"

# Data Augmentation to simulate a robust medical dataset
datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=(224, 224), batch_size=16,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=(224, 224), batch_size=16,
    class_mode='categorical', subset='validation'
)

# Build Model Base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained features

# Add Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)  # 4 Stages of Alzheimer's Risk

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

print("Fine-tuning model classification layers...")
model.fit(train_gen, validation_data=val_gen, epochs=5)

model.save("ai_engine/deepretina_model.h5")
print("SUCCESS: Model weights saved completely as deepretina_model.h5")