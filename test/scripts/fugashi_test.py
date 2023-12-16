import spacy
from fugashi import Tagger

def extract_nouns_adjs(text):
    words = []
    doc = nlp(text)
    for token in doc:
        if token.pos_ in ['NOUN', 'ADJ']:
            words.append(token.text)
    return " ".join(words)

nlp = spacy.load('ja_ginza_electra')
sample_txt = '私が最近見た映画は、約束のネバーランドでした。スターバックス鬼滅の刃コラボ。スタバはインスタ映え。'
doc = nlp(sample_txt)

print('------')
print(doc)

print('------')
print([token.text for token in doc if token.pos_ in ['NOUN','ADJ']])


print('------')
print(extract_nouns_adjs(sample_txt))

print('------')
for token in doc:
    print(token.text, token.pos_, token.tag_, token.dep_)

print('------')
tagger = Tagger()
print(tagger.parse(sample_txt))