#기계학습 라이브러리를 활용하여 선형회귀 구현시 사이킥런 사용
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

data_home = 'http://github.com/dknife/ML/raw/main/data/'
#데이터 파일이름을 가지고와 읽어라
lin_data = pd.read_csv(data_home+'pollution.csv')
x = lin_data['input'].to_numpy()
y = lin_data['pollution'].to_numpy()
x = x[:, np.newaxis] #선형회귀 모델의 입력형식에 맞게 차원을 증가

#최적의 파라미터를 찾는 수식 = 정규방정식

X = np.c_[np.ones((100,1)), x] # 편향을 파라미터로 다루기
#정규 방정식 풀리
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
#파라미터를 적용한 가설 함수를 이용하여 직선을 그려보자
def h(x, theta):
    return x*theta[1] + theta[0]
lin_data.plot(kind = 'scatter', x= 'input', y = 'pollution')
plt.plot([0, 1], [h(0, theta), h(1, theta)])
