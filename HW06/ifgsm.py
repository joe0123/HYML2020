import os
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np

import torch
# Loss function
import torch.nn.functional as F
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
seed = 1114
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda")



"""# 讀取資料庫"""

# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200



class Attacker:
    def __init__(self, img_dir, label, attack_dir):
        # 讀入預訓練模型 vgg16
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(os.path.join(sys.argv[1], "images"), label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)
        self.atk_dir = attack_dir

    def ifgsm_attack(self, data, target, epsilon, t):
        for i in range(t):
            data.requires_grad = True
            output = self.model(data)
            pred_y = output.max(1, keepdim=True)[1]
            if pred_y.item() != target.item():
                break
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            # 找出 gradient 的方向
            sign_data_grad = data_grad.sign()
            # 將圖片加上 gradient 方向乘上 epsilon 的 noise
            data = (data + epsilon * sign_data_grad).detach()
            #image = perturbed_image.detach()
            #image.requires_grad = True
        return data
    
    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        fail, success = 0, 0
        for i, (data, target) in enumerate(self.loader):
            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            
            perturbed_data = self.ifgsm_attack(data, target, epsilon, 20)

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
          
            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                img = np.clip(np.transpose(data_raw, [1, 2, 0]), 0, 1) * 255
                img = Image.fromarray(img.astype(np.uint8), "RGB")
                img.save(os.path.join(self.atk_dir, "{:03d}.png".format(i)))
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                img = np.clip(np.transpose(adv_ex, [1, 2, 0]), 0, 1) * 255
                img = Image.fromarray(img.astype(np.uint8), "RGB")
                img.save(os.path.join(self.atk_dir, "{:03d}.png".format(i)))
        final_acc = (fail / (success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return adv_examples, final_acc

"""# 執行攻擊 並顯示攻擊成功率"""

if __name__ == '__main__':
    # 讀入圖片相對應的 label
    df = pd.read_csv(os.path.join(sys.argv[1], "labels.csv"))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join(sys.argv[1], "categories.csv"))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    attacker = Attacker(os.path.join(sys.argv[1], "images"), df, sys.argv[2])
    # 要嘗試的 epsilon
    epsilons = [0.01]

    accuracies, examples = [], []

    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        ex, acc = attacker.attack(eps)
        accuracies.append(acc)
        examples.append(ex)

    exit()
"""# 顯示 FGSM 產生的圖片"""

cnt = 0
plt.figure(figsize=(30, 30))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,orig_img, ex = examples[i][j]
        # plt.title("{} -> {}".format(orig, adv))
        plt.title("original: {}".format(label_name[orig].split(',')[0]))
        orig_img = np.transpose(orig_img, (1, 2, 0))
        plt.imshow(orig_img)
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
        plt.title("adversarial: {}".format(label_name[adv].split(',')[0]))
        ex = np.transpose(ex, (1, 2, 0))
        plt.imshow(ex)
plt.tight_layout()
plt.savefig("my_attack.jpg")

