with open("zh_en_data/train.zh") as f:
    data = str(f.read())

sentence_1 = "几乎已经没有地方容纳这些人,资源已经用尽"
for char in sentence_1:
    print(char, data.count(char))

print()

for i in range(0, len(sentence_1) - 1, 1):
    char = sentence_1[i : i + 2]
    print(char, data.count(char))

print()

for i in range(0, len(sentence_1) - 2, 1):
    char = sentence_1[i : i + 3]
    print(char, data.count(char))
