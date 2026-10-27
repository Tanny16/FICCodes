import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import time
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import gc
import shutil

class StockDataset_by_year(Dataset):
    # task任务类型，默认为回归
    def __init__(self, base_dir, start_year, end_year, task='regression'):
        self.base_dir = base_dir
        self.years = list(range(start_year, end_year + 1))
        self.task = task

        self.data_full = []
        self.features_list = []
        self.labels_list = []

        # 加载数据并使用进度条
        for year in tqdm(self.years, desc="Loading Data by Year"):
            data_file = os.path.join(base_dir, f"{year}_df_data.feather")
            label_file = os.path.join(base_dir, f"{year}_real_RET.feather")

            # 读取数据和标签
            X = pd.read_feather(data_file)
            y = pd.read_feather(label_file)
            print(f"Loaded data for year {year}:")

            # 如果任务是分类，则将y的值转化为0或1
            if self.task == 'classification':
                y['real_RET'] = (y['real_RET'] > 0).astype(float)

            combined_df = pd.concat([X, y['real_RET']], axis=1)

            reserved_cols = ['permno', 'DATE', 'mvel1', 'real_RET']
            feature_cols = [col for col in X.columns if col not in ['permno', 'DATE']]

            self.data_full.append(combined_df[reserved_cols + feature_cols])

            self.features_list.append(combined_df[feature_cols].values)
            self.labels_list.append(combined_df['real_RET'].values)

        self.data_full = pd.concat(self.data_full, ignore_index=True)
        features_all = np.vstack(self.features_list)
        labels_all = np.hstack(self.labels_list)

        self.features = torch.tensor(features_all, dtype=torch.float32)
        self.labels = torch.tensor(labels_all, dtype=torch.float32)

        print(f"Data loaded from {start_year} to {end_year}.")
        print(f"Total samples: {len(self.labels)}")
        print(f"Feature dimension: {self.features.shape[1]}")
        print(f"Task: {self.task}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.features[idx]
        y = self.labels[idx]
        return X, y, idx


class ModelRunnerByYear:
    """
    一个可复用的类，使用以加载的模型对验证集进行预测，将预测结果输出为CSV
    可将任意PyTorch模型实例传入run()方法进行训练和验证集预测。
    """

    def __init__(self,
                 base_dir,
                 val_start_year=1987,
                 val_end_year=2016,
                 output_csv="predictions_only.csv",
                 batch_size=64,
                 task='regression'):
        self.base_dir = base_dir
        self.val_start_year = val_start_year
        self.val_end_year = val_end_year
        self.output_csv = output_csv
        self.batch_size = batch_size
        self.task = task

        print('Loading validation dataset...')
        self.val_dataset = StockDataset_by_year(self.base_dir, self.val_start_year, self.val_end_year, task=self.task)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f'Prediction device:{self.device}')

    def run(self, model):
        model.to(self.device)
        model.eval()

        softmax = torch.nn.Softmax(dim=1)
        predictions = []
        indices = []
        with torch.no_grad():
            for X_batch, y_batch, idx_batch in tqdm(self.val_dataloader, desc=f"Valid"):
                X_batch = X_batch.to(self.device)
                preds = model(X_batch).squeeze()  # 回归输出单值
                preds = preds.cpu().numpy()
                predictions.extend(preds)
                indices.extend(idx_batch.cpu().numpy())

        val_data_full = self.val_dataset.data_full.iloc[indices].copy()

        val_data_full.reset_index(drop=True, inplace=True)

        val_data_full['up_prob'] = predictions


        final_df = val_data_full[[ 'permno', 'DATE', 'up_prob', 'mvel1' 'real_RET']].copy()
        final_df.rename(columns={
            'permno': 'StockID',
            'DATE': 'ending_date',
            'mvel1': 'MarketCap',
            'real_RET': 'ret_val'
        }, inplace=True)

        final_df.to_csv(self.output_csv, index=False)
        print(f"预测结果已保存至 {self.output_csv}，格式如下：")
        print(final_df.head())



# 用NN1进行验证
class NN1(nn.Module):
    def __init__(self, input_dim, hidden_neurons=128):  # 添加 hidden_neurons 参数
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_neurons)  # 使用 hidden_neurons 作为隐藏层大小
        self.out = nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.out(x)


# 使用示例


if __name__ == "__main__":
    input_dim = 87
    model = NN1(input_dim=input_dim, hidden_neurons=128)

    # 加载事先训练好的最佳模型权重
    model.load_state_dict(torch.load('data/resources/NN1_Run2_Epochs50_LR0.001_Patience10_R2_-0.032997.pth'))

    predictor = ModelRunnerByYear(
        base_dir='data/processed/yearly',
        val_start_year=1987,
        val_end_year=2016,
        output_csv="NN1_best_predictions.csv",
        batch_size=64,
        task='regression'
    )

    predictor.run(model)
