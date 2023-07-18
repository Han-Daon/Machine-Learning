import matplotlib.pyplot as plt
import pandas as pd

#데이터 저장 위치
data_home = 'http://github.com/dknife/ML/raw/main/data/'
#데이터 파일이름을 가지고와 읽어라
lin_data = pd.read_csv(data_home+'pollution.csv')
print(lin_data)


#읽어들인 데이터를 시각적으로 확인
lin_data.plot(kind = 'scatter', x= 'input', y = 'pollution')

#가설을 설정해 보자. (가설은 y=wx+b라는 직선을 따를것이다!)
w,b = 1,1
x0, x1 = 0.0, 1.0
#가설의 따라 값을 계산시키자
def h(x, w, b):
    return w*x+b

#그럼 이제 가설과 비교를 해보자
lin_data.plot(kind = 'scatter', x= 'input', y = 'pollution')
plt.plot([x0, x1], [h(x0, w, b), h(x1, w, b)])

#좋은 가설이라면 데이터가 직선 근처에 있어야함
#평균 제곱오차는 기계학습에서 가장 흔히 사용되는 오차척도
import numpy as np 
y_hat = np.array([1.2, 2.1, 2.9, 4.1, 6.3, 7.1, 7.7, 8.5, 10.1])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, ,9 , 10])
diff_square  = (y_hat - y)**2
e_mes = diff_square.sum() / len(y)

from sklearn.matrics import mean_squared_error
print('Mean squared error'), mean_squared_error(y_hat, y)


#오차로 가설을 평가하고 좋은 가설 찾기
w, b = -3, 6
x = lin_data['input'].to_numpy()
y = lin_data['pollution'].to_numpy()
y_pred = h(x, w, b)
error = (y_pred - y)
error

#최소제곱법 : 기울기가 극소인 지점을 찾는것
#(w,b)벡터를 조금 옮겨주면 최적의 w와 b에 가까워질것
learning_rate = 0.005 #조금의 값
w = w - learning_rate*(error * x).sum()
b = b - learning_rate*error.sum()

lin_data.plot(kind = 'scatter', x= 'input', y = 'pollution')
plt.plot([x0, x1], [h(x0, w, b), h(x1, w, b)])
