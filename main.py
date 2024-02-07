import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

# Инициализируем модуль nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Dataset2class(torch.utils.data.Dataset):
    def __init__(self,path:str):
        super().__init__()
        self.data = pd.read_csv(path, encoding='latin-1')
        self.data = self.data.drop(columns=self.data.iloc[:, 2:6])
        self.data["v3"] = self.data["v2"].apply(self.preprocess_text)
        self.data["v1"] = LabelEncoder().fit_transform(self.data["v1"])
    # Пример метода для предобработки слов
    def preprocess_text(self,text):
        # Приведение текста к нижнему регистру


        # Токенизация текста (разбиение на слова)
        words = word_tokenize(text)

        # Удаление пунктуации
        words = [word for word in words if word not in string.punctuation]

        # Удаление стоп-слов (например, "the", "is", "and" и т.д.)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Стемминг (приведение слова к его основе)
        #porter = PorterStemmer()
        #words = [porter.stem(word) for word in words]

        # Лемматизация (приведение слова к его базовой форме)
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        text_strings = ["".join(tokens) for tokens in words]
        return text_strings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx]["v3"], self.data.loc[idx]["v1"]


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                          nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.layer_dim, batch_size, self.hidden_dim))
        x = x.unsqueeze(1)

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def main():
    train_data = Dataset2class("train.csv")
    test_data = Dataset2class("test.csv")
    X_train = train_data.data["v3"].tolist()
    X_train = [' '.join(words) for words in X_train]
    Y_train = train_data.data["v1"].tolist()
    tfidf_vectorizer = TfidfVectorizer()

    X_test = test_data.data["v3"].tolist()
    X_test = [' '.join(words) for words in X_test]
    Y_test = test_data.data["v1"].tolist()

    tfidf_matrix_test = tfidf_vectorizer.fit_transform(X_test)
    # Преобразование текстовых данных в TF-IDF матрицу
    tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)
    #X_test_tfidf = tfidf_vectorizer.transform(X_test)
    x_train = torch.tensor(tfidf_matrix.todense(), dtype=torch.float32)
    y_train = torch.tensor(Y_train, dtype=torch.long)
    train = TensorDataset(x_train, y_train)
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
    x_test = torch.tensor(tfidf_matrix_test.todense(), dtype=torch.float32, device=device)
    y_test = torch.tensor(Y_test, dtype=torch.long, device=device)
    test = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train, batch_size=16, shuffle=False)
    test_loader = DataLoader(test, batch_size=16, shuffle=False)

    # Create RNN
    input_dim = 7849  # input dimension
    hidden_dim = 200  # hidden layer dimension
    layer_dim = 5  # number of hidden layers
    output_dim = 2  # output dimension
    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=0.03)
    epochs = 50
    history = []
    for epoch in range(epochs):
        loss_val = 0
        correct_predictions = 0
        total_samples = 0

        for sample in train_loader:
            data, value = sample[0], sample[1]
            optimizer.zero_grad()
            pred = model(data)
            loss = error(pred, value)
            loss.backward()

            loss_item = loss.item()
            loss_val += loss_item

            optimizer.step()

            _, predicted = torch.max(pred.data, 1)
            total_samples += value.size(0)
            correct_predictions += (predicted == value).sum().item()

        accuracy = correct_predictions / total_samples
        history.append(loss_val / len(train_loader))
        print(f"Эпоха {epoch}")
        print(f"Loss {loss_val / len(train_loader)}")
        print(f"Точность предсказания: {accuracy * 100:.2f}%")
        print()



if __name__ == "__main__":
    main()

