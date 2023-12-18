import MeCab
import joblib
from sklearn.preprocessing import LabelEncoder

mecab = MeCab.Tagger('-d "C:/Program Files/MeCab/dic/ipadic" -u "C:/Program Files/MeCab/dic/NEologd/NEologd.dic"')

def extract_nouns_adjs(text):
    node = mecab.parseToNode(text)
    words = []
    while node:
        if node.feature.split(',')[0] in ['名詞', '形容詞']:
            words.append(node.surface)
        node = node.next
    return " ".join(words)

# Load trained model
model_path = './model/tfidf/yahoo_news_tfidf_naive_bayes_model.pkl'
model = joblib.load(model_path)

le = LabelEncoder()

categories = ['IT・科学', 'エンタメ', 'スポーツ', 'ライフ', '国内', '国際', '地域', '経済']
le.fit(categories)

# text to predict
text_to_predict = "米大リーグの記者投票によるMVPが16日（日本時間17日）、発表され、ア・リーグは大谷翔平選手（エンゼルス）が2年ぶり2度目の受賞。1931年に創設され、93年の歴史を持つ同賞では史上初となる2度目の満票選出となった。日本選手では01年にイチロー（マリナーズ）が受賞しているが、2度は初めての偉業だ。"

# Preprocessing text
preprocessed_text = extract_nouns_adjs(text_to_predict)
print(preprocessed_text)

# Category prediction
predicted_label_num = model.predict([preprocessed_text])[0]
print(predicted_label_num)
predicted_label = le.inverse_transform([predicted_label_num])[0]
print(f"Predicted category: {predicted_label}")
