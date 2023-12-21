from fugashi import Tagger
import MeCab

# def extract_nouns_adjs(text):
#     words = []
#     doc = nlp(text)
#     for token in doc:
#         if token.pos_ in ['NOUN', 'ADJ']:
#             words.append(token.text)
#     return " ".join(words)

# nlp = spacy.load('ja_ginza_electra')
# doc = nlp(sample_txt)

# print('------')
# print(f'doc: {doc}')
# print([token.text for token in doc if token.pos_ in ['NOUN','ADJ']])
# print(f'extract_nouns_adjs(sample_txt):{extract_nouns_adjs(sample_txt)}')
# print('------')

# for token in doc:
#     print(token.text, token.pos_, token.tag_, token.dep_)

sample_txt = 'スターバックスはスタバです。鬼滅の刃が上映中。すもももももももものうち。コロナ禍で大変です'

# Specify the path of Neologd
neologd_path = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'

print('------')
tagger = Tagger()
print(f'tagger.parse(sample_txt): fugashi test\n{tagger.parse(sample_txt)}')

print('------')
mecab = MeCab.Tagger('-d' + neologd_path)
print(f'tagger.parse(sample_txt): MeCab test\n{mecab.parse(sample_txt)}')