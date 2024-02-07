import json
data = [12, 1, 13]
with open("file.json", 'w') as f:
    json.dump(data, f)
with open("file.json", "r") as f:
    a = json.load(f)
print(a)
