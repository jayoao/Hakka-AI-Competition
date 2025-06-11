import os
from model_predict import predict_image
from model_train import train_model

MODEL_PATH = 'hakka_model.h5'

def main():
    # Step 1: 檢查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print("⚠️ 尚未找到訓練好的模型 hakka_model.h5")
        choice = input("是否要立即開始訓練模型？（Y/N）：").strip().lower()
        if choice == 'y':
            train_model(save_path=MODEL_PATH)
        else:
            print("❌ 請先完成模型訓練，再執行預測功能。程式結束。")
            return
    
    # Step 2: 使用者輸入圖片 → 預測
    img_path = input("請輸入要預測的圖片路徑：").strip()
    if not os.path.exists(img_path):
        print("❌ 找不到圖片，請確認路徑正確。")
        return

    predict_image(img_path)

if __name__ == '__main__':
    main()
