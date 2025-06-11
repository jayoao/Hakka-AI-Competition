from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# 載入標籤對應表
with open('hakka_labels.json', 'r', encoding='utf-8') as f:
    label_map = json.load(f)

model = load_model('hakka_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    confidence = prediction[0][pred_index]
    label_name = label_map[str(pred_index)]

    print(f"🖼️ 預測結果：{label_name}（信心值：{confidence:.2f}）")
