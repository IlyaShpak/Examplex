import torch
from simple_task import MyModel, MNIST
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from tqdm import tqdm

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy:.2%}")

def main():
    model = MyModel()
    model.load_state_dict(torch.load("my_model.pth"))
    data_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(0.5, 0.5)
    ])
    root = "./"
    test_data = MNIST(root, train=False, transform=data_tfs, download=True)
    test_loader = DataLoader(test_data, batch_size=128, drop_last=True)
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
    model.to(device)
    model.eval()
    acc = 0
    batches = 0

    for x_batch, y_batch in test_loader:
        # загружаем батч данных (вытянутый в линию)
        batches += 1
        x_batch = x_batch.view(x_batch.shape[0], -1).to(device)
        y_batch = y_batch.to(device)

        preds = torch.argmax(model(x_batch), dim=1)
        acc += (preds == y_batch).cpu().numpy().mean()

    print(f'Test accuracy {acc / batches:.3}')



if __name__ == "__main__":
    main()