# IDF (Inverse Document Frequency) 실습 예제

## 개요
IDF(Inverse Document Frequency)는 자연어 처리 및 정보 검색에서 등장한 개념으로, 문서 집합에서 특정 단어의 중요성을 평가하는 데 사용됩니다. 특정 단어가 드물게 나타날수록, 그 단어의 IDF 값은 높아지며, 이는 해당 단어가 특정 문서를 구별하는 데 유용하다는 것을 의미합니다.

IDF는 TF(Term Frequency)와 IDF를 곱한 값인 TF-IDF를 구하기 위해 사용됩니다. 
TF는 각 문서에 대한 BoW를 하나의 행렬로 만든 DTM(Document-Term Matrix)에서 각 단어들이 가진 값을 의미합니다.
TF-IDF는 각 단어의 중요성을 반영하여, 많은 경우에서 기존 DTM을 사용하는 것보다 좋은 성능을 얻을 수 있습니다.

## IDF 계산 공식
IDF의 계산 공식은 다음과 같습니다:<br>
$ idf(t) = \log(N/df(t)+1)$<br>
여기서:
- $N$은 문서 집합에 있는 총 문서의 수
- $df(t)$는 단어 $t$가 포함된 문서의 수
- 로그 함수는 값의 스케일을 변환 하여, 큰 값들이 너무 지배적이지 않도록 조정합니다.

## Python을 사용한 IDF 계산 예제
아래 예제에서는 Python을 사용하여 "cat"이라는 단어의 IDF 값을 계산합니다.

```python
import math
from collections import Counter

# 문서 집합 예시
documents = [
    "the cat is on the table",
    "the dog is in the house",
    "the cat and the dog are friends"
]

# 특정 단어의 문서 빈도수를 계산하는 함수
def compute_document_frequency(word, documents):
    return sum(1 for doc in documents if word in doc.split())

# IDF 값을 계산하는 함수
def compute_idf(word, documents):
    N = len(documents)
    df = compute_document_frequency(word, documents)
    return math.log(N / (1 + df))

# 특정 단어에 대한 IDF 값 계산
word = "cat"
idf_value = compute_idf(word, documents)
idf_value
```

이 예제에서 "cat" 단어의 IDF 값은 0.0으로 계산되었습니다. 이는 "cat"이 문서 집합에서 비교적 자주 나타나는 단어임을 의미하며, 낮은 중요도를 갖는다고 할 수 있습니다.

TF-IDF를 계산 할때 이처럼 IDF의 값이 0이 나오는 것을 방지하기 위하여 idf 결과값에 1을 더하여 사용하기도 합니다. 
이는 idf 가 0이 되면 tf 가 0인 것과 tf-idf 값이 구분되지 않기 때문입니다.

## scikit-learn의 TfidfVectorizer 예시 코드

사이킷런 라이브러리의 TfidfVectorizer 클래스에서는 IDF를 다음과 같이 계산합니다.
원래의 식에서 분자에도 1을 더해주었습니다. 
$idf(t) = \log(N+1/df(t)+1) + 1$

```python
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
```
## Output
```
       Word  IDF Value
0       and   1.693147
1       are   1.693147
2       cat   1.287682
3       dog   1.287682
4   friends   1.693147
5     house   1.693147
6        in   1.693147
7        is   1.287682
8        on   1.693147
9     table   1.693147
10      the   1.000000
```