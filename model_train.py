from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def train_model(data_dir='data/', save_path='hakka_model.h5', num_classes=3):
    img_size = (224, 224)
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = train_datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size,
                                                  class_mode='categorical', subset='training')
    val_gen = train_datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size,
                                                class_mode='categorical', subset='validation')

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save(save_path)
    print(f"✅ 模型已儲存至 {save_path}")

# 主動呼叫訓練
if __name__ == '__main__':
    train_model()
