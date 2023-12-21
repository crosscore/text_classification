from janome.tokenizer import Tokenizer

t = Tokenizer()
for token in t.tokenize("スターバックスの略はスタバだよ"):
    print(token)
# for token in t.tokenize("私が最近見た映画は、約束のネバーランドでした。スターバックス鬼滅の刃コラボ。スタバはインスタ映え。"):
#     print(token)
