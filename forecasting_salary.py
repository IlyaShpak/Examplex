import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.df = pd.read_csv(path)
        self.df["experience_level"] = self.df["experience_level"].map({"SE": 0, "MI": 1, "EN": 2, "EX": 3})
        self.df["employment_type"] = self.df["employment_type"].map({"FT": 0, "CT": 1, "FL": 2, "PT": 3})
        self.df["job_title"] = self.df["job_title"].factorize()[0]
        self.df["employee_residence"] = self.df["employee_residence"].factorize()[0]
        self.df["company_location"] = self.df["company_location"].factorize()[0]
        self.df["company_size"] = self.df["company_size"].factorize()[0]
        self.df = self.df.drop("salary", axis=1)
        self.df = self.df.drop("salary_currency", axis=1)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target = self.df.loc[idx]["salary_in_usd"]
        df = np.delete(self.df.loc[idx].values, 4)
        return df, target


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.funct = nn.LeakyReLU()
        self.lc1 = nn.Linear(8, 32, dtype=torch.float32)
        self.lc2 = nn.Linear(32, 32, dtype=torch.float32)
        self.lc3 = nn.Linear(32, 1, dtype=torch.float32)

    def forward(self, x):
        out = self.lc1(x)
        out = self.funct(out)
        out = self.lc2(out)
        out = self.funct(out)
        out = self.lc3(out)
        return out


def accuracy(true, predicted):
    return torch.mean(torch.abs(predicted - true))


def main():
    train_data = Dataset2class("train.csv")
    test_data = Dataset2class("test.csv")
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True,
        batch_size=batch_size, num_workers=1, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, shuffle=True,
        batch_size=batch_size, num_workers=1, drop_last=False
    )
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
    model = MyNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=0.03)
    epochs = 50
    history = []
    for epoch in range(epochs):
        loss_val = 0
        acc_val = 0
        for sample in train_loader:
            data, value = sample[0], sample[1]
            optimizer.zero_grad()
            data = torch.tensor(data, dtype=torch.float32, device=device)
            value = torch.tensor(value, dtype=torch.float32, device=device).unsqueeze(1)
            pred = model(data)
            loss = loss_fn(pred, value)
            loss.backward()

            loss_item = loss.item()
            loss_val += loss_item

            optimizer.step()
            acc_current = accuracy(pred, value)
            acc_val += acc_current
        history.append(loss_val / len(train_loader))
        print(f"Эпоха {epoch}")
        print(f"Loss {loss_val/len(train_loader)}")
        print(f"Acc {acc_val/len(train_loader)}")
        print()
    with open("test.json", "w") as file:
        json.dump(history, file)
    plt.plot()
    plt.show()


if __name__ == "__main__":
    main()
