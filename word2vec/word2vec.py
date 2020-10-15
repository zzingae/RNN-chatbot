# https://wikidocs.net/50739 참조

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt # Okt는 원래 이름이 Twitter였으나 0.5.0 버전 이후부터 이름이 Okt 바뀌었다.
import numpy as np
import json
from gensim.models import Word2Vec


# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
train_data = pd.read_table('ratings.txt')

if train_data.isnull().values.any(): # Null?
    train_data = train_data.dropna(how = 'any')

print(len(train_data)) # 리뷰 개수 출력

# 정규 표현식을 통한 한글 외 문자 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")   
# 불용어 정의
# stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
stopwords=[]

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()
tokenized_data = []
count=0
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_X)
    count+=1
    print(count)


model = Word2Vec(sentences = tokenized_data, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

np.save('naver_movie_embed.npy', model.wv.vectors)

vocab={}
for i,word in enumerate(model.wv.index2word):
    vocab[word]=i
with open('vocabulary.json','w') as f:
    json.dump(vocab,f,ensure_ascii=False)
