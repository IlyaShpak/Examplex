import torch.nn as nn
import torch
from matplotlib import pyplot as plt


def train_model():
    criterion = nn.MSELoss()
    batch_size = 10
    epochs = 100
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Sigmoid()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, betas=(0.9, 0.99))
    x = torch.FloatTensor([i for i in range(100)])
    y = torch.sin(x)
    history = []
    for i in range(epochs):
        for j in range(0, len(x)):
            logits = model(x[j].unsqueeze(0))
            loss = criterion(logits, y[j])
            history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    x_test = torch.FloatTensor([i for i in range(100, 200)])
    y_test = torch.sin(x_test)
    plt.plot([i for i in range(10000)], history)
    model.eval()
    with torch.no_grad():
        for k in range(len(x_test)):
            outputs = model(x_test[k].unsqueeze(0))
            loss = criterion(outputs, y_test[k])
            print(loss.item())
    print(outputs, y_test)


def main():
    train_model()


if __name__ == "__main__":
    main()
