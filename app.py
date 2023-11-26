import streamlit as st
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import numpy as np
import model
import matplotlib.pyplot as plt

# MTCNNの初期化
mtcnn = MTCNN(keep_all=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

# 顔検出機能の定義
def detect_faces(image):
    # PILイメージを配列に変換
    image_array = np.array(image)
    
    # MTCNNを使用して顔を検出
    boxes, _ = mtcnn.detect(image_array)
    
    # 検出された顔の領域をリストとして返す
    face_images = []
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box
            face = image.crop((x, y, w, h))
            face_images.append(face)
    
    return face_images

st.title("日向坂メンバー認識アプリ")

# 画像アップローダーの配置
uploaded_file = st.sidebar.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # アップロードされた画像を表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 顔検出を実行
    face_images = detect_faces(image)

    if face_images:
        for face_img in face_images:
            # 画像の前処理
            tensor_image = model.transform_image(face_img).unsqueeze(0)

            # モデルで予測
            top_probs, top_classes = model.predict_with_top_classes(tensor_image, n=5)

            # 確率の閾値を設定（ここでは例として1e-4を使用）
            threshold = 1e-4
            
            # トップ5の予測結果をフィルタリング（0%の結果を排除）
            filtered_probs = [(prob, class_idx) for prob, class_idx in zip(top_probs, top_classes) if prob > 0]

            if filtered_probs:
                # 最も確率が高い予測のラベル
                prediction_label = filtered_probs[0][1]
                st.write(f"予測結果: {model.memberHNZ[prediction_label]}")

                # フィルタリングされた予測結果を箇条書きで表示
                st.write("トップ5の予測結果:")
                for i, (prob, class_idx) in enumerate(filtered_probs):
                    st.write(f"{i+1}. {model.memberHNZ[class_idx]}")

                # 予測結果の円グラフをプロット
                fig, ax = plt.subplots()
                filtered_top_probs = [prob for prob, _ in filtered_probs]
                model.plot_pie_chart(filtered_top_probs, [class_idx for _, class_idx in filtered_probs], ax=ax)
                st.pyplot(fig)
            else:
                st.write("顔認識の結果が不確実です。")

    else:
        st.write("顔が検出されませんでした。別の画像を試してください。")
