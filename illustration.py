import json
import matplotlib.pyplot as plt

def main():
    with open("test.json", "r") as file:
        data = json.load(file)
    plt.plot(data)
    plt.show()

if __name__ == "__main__":
    main()
