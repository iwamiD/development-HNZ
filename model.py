import torch
import torchvision.transforms as transforms
import os

import matplotlib as mpl

import torch.nn as nn
import torchvision.models as models

# 日本語フォントのパスを指定する
# font_path = 'C:/Windows/Fonts/meiryo.ttc'  # あなたの環境に合わせて変更してください

# 日本語フォントの設定
mpl.rcParams['font.family'] = 'Meiryo'  # あなたの環境に合わせてフォント名を変更してください
mpl.rcParams['axes.unicode_minus'] = False  # 日本語フォント使用時のマイナス記号対策

# クラス名のマッピング
memberHNZ = [
    "潮紗理菜", "影山優佳", "加藤史帆", "斎藤京子", "佐々木久美", "佐々木美鈴", "高瀬愛奈", "高本彩花", "東村芽依", 
    "金村美玖", "河田陽菜", "小坂奈緒", "富田鈴花", "丹生明里", "濱岸ひより", "松田好花", "宮田愛萌", "渡邉美穂",
    "上村ひなの", "高橋未来虹", "森本茉莉", "山口陽世", 
    "石塚瑶季", "岸帆夏", "小西夏菜実", "清水理央", "正源司陽子", "竹内希来里", 
    "平尾帆夏", "平岡海月", "藤嶌果歩", "宮地すみれ", "山下葉留花", "渡辺莉奈"
]

class HNZResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(HNZResNet, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        in_features = resnet34.fc.in_features
        # ドロップアウト層を追加
        self.dropout = nn.Dropout(dropout_rate)
        resnet34.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, num_classes)
        )
        self.resnet34 = resnet34

    def forward(self, x):
        return self.resnet34(x)

# GPUが利用可能ならばGPUを、そうでなければCPUを使用する
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# モデルの初期化
model = HNZResNet(len(memberHNZ)).to(device)

# state_dictのみをロード
# モデルのロード時のエラーハンドリング
model_path = './MachineLearning/HNZrecognition/HNZRecmodel.pth'
if not os.path.exists(model_path):
    raise ValueError(f"Model file {model_path} not found.")
try:
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
except Exception as e:
    raise ValueError(f"Error loading model from {model_path}: {e}")

# モデルを評価モードに設定
model.eval()

# 画像の前処理

# 正規化の平均と標準偏差
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # トレーニング時と同じサイズにリサイズ
    transforms.ToTensor(),          # PILイメージまたはNumPy ndarrayをTensorに変換
    # transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),  # トレーニング時と同じ正規化
])

def transform_image(image):
    """ 画像の前処理関数 """
    return transform(image)

# 予測処理の関数
def predict_with_top_classes(image_tensor, n=5):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, n)
    return top_probs.squeeze().tolist(), top_classes.squeeze().tolist()
    
def plot_pie_chart(top_probs, top_classes, threshold=0.01, ax=None):
    """ 
        トップクラスの予測確率を円グラフとしてプロット。
        予測確率が指定された閾値（デフォルトは1%）より高いもののみを含む。
    """
    # 0%の予測を除外
    filtered_probs_classes = [(prob, cls) for prob, cls in zip(top_probs, top_classes) if prob > threshold]
    if not filtered_probs_classes:
        raise ValueError("No class predictions exceed the threshold. Can't plot pie chart.")

    # 解体されたリストを作成
    filtered_probs, filtered_classes = zip(*filtered_probs_classes)

    # ラベルをフィルターされたクラスのみに対応させる
    labels = [memberHNZ[i] for i in filtered_classes]
    ax.pie(filtered_probs, labels=labels, autopct='%1.1f%%', startangle=90)
    # タイトルに日本語フォントを適用
    ax.set_title('Top predictions', fontname='Meiryo')  # あなたの環境に合わせてフォント名を変更してください
    ax.axis('equal')