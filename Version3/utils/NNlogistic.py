import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# 自定义 Logistic Regression
class LogisticRegression(pl.LightningModule):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(input_dim, num_classes))
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))
        self.criterion = torch.nn.CrossEntropyLoss()  # 对于多分类任务

    def forward(self, x):
        logits = x @ self.weights + self.bias
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算预测
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # 返回验证损失和准确率
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        # 汇总所有 batch 的准确率
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("avg_val_acc", avg_acc, prog_bar=True)
        print(f"Validation Accuracy: {avg_acc:.4f}")

    def configure_optimizers(self):
        return torch.optim.SGD([self.weights, self.bias], lr=1e-3)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=32):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def setup(self, stage=None):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2)
        self.train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        self.val_data = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
# Usage
# from utils.NNlogistic import LogisticRegression, CustomDataModule
# import pytorch_lightning as pl

# dm = CustomDataModule(full_data_X.to_numpy(), full_data_y.to_numpy(), batch_size=16)

# input_dim = full_data_X.shape[1]

# model = LogisticRegression(input_dim=input_dim, num_classes=2)

# trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1)

# trainer.fit(model, datamodule=dm)