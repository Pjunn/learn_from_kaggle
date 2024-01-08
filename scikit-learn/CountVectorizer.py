''' 
텍스트 데이터에 대해서 BOW를 만들기 위해 사용할 수 있는 CountVectorizer 클래스입니다.
텍스트를 이루는 단어가 나타나는 횟수로 벡터를 만듭니다.

ngram_range 파라미터는 단어와 단어의 연속된 조합(N-gram)또한 반영하여 벡터를 생성할 수 있도록 합니다.
만약 ngram_range=(1, 2)일때, 텍스트를 이루는 단어 하나 하나가 카운팅 되고 연속된 두개의 단어 조합이 추가적으로 카운팅되어 벡터에 반영합니다.    
'''
from sklearn.feature_extraction.text import CountVectorizer

# 샘플 텍스트 데이터
text_data = [
    "I love writing code in Python",
    "Python is a great language for data analysis",
    "Data science is fun"
]

# CountVectorizer 초기화
vectorizer = CountVectorizer(ngram_range=(1, 2))

# 텍스트 데이터에 fit하고 변환하기
vectorized_data = vectorizer.fit_transform(text_data)

# 결과 출력
print("Feature Names:", vectorizer.get_feature_names_out())
print("Vectorized Data:\n", vectorized_data.toarray())