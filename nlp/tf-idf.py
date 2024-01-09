import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 문서 집합
documents = [
    "the cat is on the table",
    "the dog is in the house",
    "the cat and the dog are friends"
]

# TF-IDF 변환기 생성 (IDF 계산을 위해 사용)
vectorizer = TfidfVectorizer()

# 문서 집합에 대한 TF-IDF 행렬 계산
tfidf_matrix = vectorizer.fit_transform(documents)

# IDF 값을 추출
idf_values = vectorizer.idf_

# 단어와 해당 IDF 값 매핑
word_idf_dict = dict(zip(vectorizer.get_feature_names_out(), idf_values))

# 결과를 판다스 데이터프레임으로 변환
idf_df = pd.DataFrame(list(word_idf_dict.items()), columns=['Word', 'IDF Value'])

print(idf_df)