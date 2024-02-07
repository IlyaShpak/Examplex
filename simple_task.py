import torch
from torchvision.datasets import MNIST
import torchvision.transforms as tfs
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_lin = nn.Linear(784, 64)
        self.s_lin = nn.Linear(64, 10)

    def forward(self, X):
        X = F.relu(self.f_lin(X))
        return self.s_lin(X)



def main():
    data_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(0.5, 0.5)
    ])
    root = "./"
    train = MNIST(root, train=True, transform=data_tfs, download=True)
    test = MNIST(root, train=False, transform=data_tfs, download=True)

    batch_size = 128

    train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, drop_last=True)

    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
    print(device)
    model = MyModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimized = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    epohs = 2
    history = []
    local_history = []
    for i in tqdm(range(epohs)):
      for x, y in train_loader:
          x = x.view(x.size(0), -1).to(device)
          y = torch.LongTensor(y).to(device)
          logits = model(x)

          loss = loss_fn(logits,y)
          history.append(loss.item())
          optimized.zero_grad()
          loss.backward()
          optimized.step()
    print(history[-1])
    plt.plot(history)
    plt.title('Training Loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.show()
    torch.save(model.state_dict(), "my_model.pth")

if __name__ == "__main__":
    main()
