import os
import warnings
from datetime import datetime

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from adabound import AdaBound
from PIL import Image
import optuna
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torchvision import models, transforms
from torchvision.transforms.functional import resize

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, default_collate

# 実行モードの設定
# 'TRAIN'で通常の学習、'TRIAL'でOptunaのトライアルを実行する
RUN_MODE = 'TRIAL'

# 余計な警告を非表示
warnings.simplefilter('ignore')

class AlbumentationsTransform:
    def __init__(self):
        self.transform = A.Compose([
            # 最初の OneOf ブロック: 主に色調や明度に関する変換
            A.OneOf([
                A.Blur(blur_limit=(2, 3), p=0.1),
                A.GaussianBlur(blur_limit=(3, 5), p=0.1),
                A.GaussNoise(var_limit=(10.0, 20.0), p=0.1),
                A.ISONoise(intensity=(0.1, 0.3), color_shift=(0.05, 0.1), p=0.1),
                A.MultiplicativeNoise(multiplier=(0.98, 1.02), p=0.1),
                A.HueSaturationValue(hue_shift_limit=(-5, 5), sat_shift_limit=(-10, 10), val_shift_limit=(-5, 5), p=0.2),
                A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10), p=0.2),
                A.RandomGamma(gamma_limit=(90, 110), p=0.2),
                A.RandomBrightnessContrast(brightness_limit=(-0.01, 0.01), contrast_limit=(-0.01, 0.01), p=0.2),
                A.CLAHE(clip_limit=2.0, p=0.1),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.2),
                A.ChannelShuffle(p=0.1),
                A.FancyPCA(alpha=0.1, p=0.1),
            ], p=0.5),

            # 2つ目の OneOf ブロック: ジオメトリック変形やテクスチャーに関わる変換
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.2),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.1),
                A.CoarseDropout(max_height=4, max_width=4, max_holes=2, p=0.1),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.1, 0.3), p=0.1),
                A.Equalize(mode='cv', by_channels=True, p=0.1),
            ], p=0.5),

            A.Resize(224, 224, interpolation=Image.BILINEAR)
        ])

    def __call__(self, data):
        image, label = data['image'], data['label']
        transformed_data = self.transform(image=np.array(image))
        transformed_image = transformed_data['image']
        return {'image': transformed_image, 'label': label}


class HNZDataset(Dataset):
    def __init__(self, data_dir, image_list=None, transform=None, class_multipliers=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
        self.class_multipliers = class_multipliers

        if image_list is not None:
            self.image_list = image_list
        else:
            self.image_list = []
            for class_name in self.classes:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        self.image_list.append(os.path.join(class_name, img_name))

    def __len__(self):
        if self.class_multipliers:
            total_len = 0
            for img_rel_path in self.image_list:
                class_name = img_rel_path.split(os.path.sep)[0]
                total_len += self.class_multipliers[class_name]
            return total_len
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        actual_idx = idx % len(self.image_list)
        img_rel_path = self.image_list[actual_idx]

        try:
            img_path = os.path.join(self.data_dir, img_rel_path)
            image = Image.open(img_path).convert("RGB")
            label = self.class_to_idx[img_rel_path.split(os.path.sep)[0]]
            
            if self.transform:
                transformed_data = self.transform({'image': image, 'label': label})
                transformed_image = transforms.ToTensor()(transformed_data['image'])  # 画像をテンソルに変換
            else:
                transformed_image = transforms.ToTensor()(image)
            
            image.close()  # 画像のメモリを解放

            return transformed_image, label
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            return None, None

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

class Trainer:
    def __init__(self, train_loader, val_loader, model, num_classes, device, trial=None):
        self.device = device
        # self.model = model(num_classes=num_classes).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optunaのトライアルをクラス内で初期化
        self.trial = trial
        if self.trial:
            self.learning_rate = self.trial.suggest_float('learning_rate', 0.00001, 0.1 , log=True)
            self.weight_decay = self.trial.suggest_float('weight_decay', 0.00000001, 0.1 , log=True)
            self.final_lr = self.trial.suggest_float('final_lr', 0.05, 0.1)
            self.step_size = self.trial.suggest_int('step_size', 5, 15)
            self.gamma = self.trial.suggest_float('gamma', 0.5, 1.0)
            self.dropout_rate = self.trial.suggest_float('dropout_rate', 0.1, 0.5)
        else:
            self.learning_rate = 0.00024465282985250105
            self.weight_decay = 0.0002481174988217314
            self.final_lr = 0.08790264561720262
            self.step_size = 6
            self.gamma = 0.6933446432473303
            self.dropout_rate = 0.41945740317867874
            
            # デフォルトパラメータ
            # self.learning_rate = 0.001
            # self.weight_decay = 0.0001
            # self.final_lr = 0.01
            # self.step_size = 10
            # self.gamma = 0.5
            # self.dropout_rate = 0.3
        
        self.model = model(num_classes=num_classes, dropout_rate=self.dropout_rate).to(self.device)
        
        # Criterion
        self.criterion = nn.CrossEntropyLoss()

        # オプティマイザー
        self.optimizer = AdaBound(self.model.parameters(), lr=self.learning_rate, final_lr=self.final_lr, weight_decay=self.weight_decay)

        # 学習率スケジューラ
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def calculate_accuracy(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def plot_metrics(self, train_losses, val_losses, train_accuracies, val_accuracies):
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # ax1に損失をプロット
        ax1.plot(train_losses, label='Training Loss', color='b')
        ax1.plot(val_losses, label='Validation Loss', color='c')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params('y', colors='b')
        ax1.legend(loc='upper left')

        # ax2はax1と同じx軸を共有するが、y軸は異なる
        ax2 = ax1.twinx()
        ax2.plot(train_accuracies, label='Training Accuracy', color='r')
        ax2.plot(val_accuracies, label='Validation Accuracy', color='m')
        ax2.set_ylabel('Accuracy (%)', color='r')
        ax2.tick_params('y', colors='r')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
    
    def train(self, num_epochs):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_accuracy = 0  # 最高の検証精度を追跡するための変数
        
        print(f'TrainingSTART_{datetime.now()}')
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            train_losses.append(train_loss)

            val_loss = self.evaluate()
            val_losses.append(val_loss)

            train_accuracy = self.calculate_accuracy(self.train_loader)
            train_accuracies.append(train_accuracy)

            val_accuracy = self.calculate_accuracy(self.val_loader)
            val_accuracies.append(val_accuracy)
            
            if epoch % 10 == 9:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
                    print(f'Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
                    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            if (RUN_MODE == 'TRAIN') & (val_accuracy > best_val_accuracy):
                    best_val_accuracy = val_accuracy
                    # 精度が改善したのでモデルを保存
                    print(f'現在の最大精度：{best_val_accuracy}')
                    self.save_model(f'./MachineLearning/HNZrecognition/model_prm/HNZRecmodel_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
                             
            self.scheduler.step()
            
        if RUN_MODE == 'TRAIN':
            self.plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
        
    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, labels).item()

        return val_loss / len(self.val_loader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def custom_collate(batch):
    # 同一サイズにリサイズ
    target_size = (224, 224)
    new_batch = []
    for item in batch:
        image, label = item
        resized_image = resize(image, target_size)
        new_batch.append((resized_image, label))
    
    return default_collate(new_batch)

def create_datasets(data_dir, image_list, class_multipliers, test_size=0.2):
    # クラスごとのインデックスを取得
    labels = [img_path.split(os.path.sep)[0] for img_path in image_list]
    class_indices = [class_to_idx[label] for label in labels]

    # 層化サンプリング
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    train_indices, val_indices = next(sss.split(image_list, class_indices))

    # 学習用と検証用のデータセットを作成
    train_dataset = HNZDataset(data_dir, [image_list[i] for i in train_indices], transform=AlbumentationsTransform(), class_multipliers=class_multipliers)
    val_dataset = HNZDataset(data_dir, [image_list[i] for i in val_indices])

    return train_dataset, val_dataset
    
def objective(trial):
    # トレーナーの初期化と学習の実行
    trainer = Trainer(train_loader, val_loader, HNZResNet, num_classes=34, device=get_device(), trial=trial)
    trainer.train(num_epochs=30)
    
    val_accuracy = trainer.calculate_accuracy(val_loader)
    return val_accuracy

if __name__ == '__main__':
        # データセットのディレクトリ
    data_dir = "./MachineLearning/HNZrecognition/data/"

    # 画像ファイルのリストを取得
    image_list = [os.path.join(class_name, img_name)
                  for class_name in os.listdir(data_dir)
                  for img_name in os.listdir(os.path.join(data_dir, class_name))
                  if os.path.isdir(os.path.join(data_dir, class_name))]
    
    # クラスごとのデータ数に応じて拡張の回数を決定
    class_counts = {}
    class_multipliers = {}

    for img_path in image_list:
        class_name = img_path.split(os.path.sep)[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    for class_name, count in class_counts.items():
        # if count < 80:
        #     multiplier = 12
        # elif count < 200:
        #     multiplier = 10
        # elif count < 350:
        #     multiplier = 8
        # else:
        #     multiplier = 6
        if count < 100:
            multiplier = 12
        else:
            multiplier = 3
            
        class_multipliers[class_name] = multiplier
        
    # クラスのインデックスを作成
    class_to_idx = {class_name: i for i, class_name in enumerate(sorted(os.listdir(data_dir)))}
    
    # データセットの準備
    # class_multipliers を利用して層化サンプリングを行う
    train_dataset, val_dataset = create_datasets(data_dir, image_list, class_multipliers, test_size=0.2)

    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=6, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False, num_workers=4, collate_fn=custom_collate)
    
    if RUN_MODE == 'TRIAL':
        study = optuna.create_study(direction='maximize') # 最大化を目指すスタディを作成
        study.optimize(objective, n_trials=30)

        # 最適なパラメータをテキストファイルとして出力
        with open('./MachineLearning/HNZrecognition/best_params.txt', 'w') as f:
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
                
    else: # RUN_MODE == 'TRAIN'
        # トレーナーの初期化と学習の実行
        trainer = Trainer(train_loader, val_loader, HNZResNet, num_classes=34, device=get_device())
        trainer.train(num_epochs=30)