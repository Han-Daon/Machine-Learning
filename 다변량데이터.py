import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns #시각화를 위해 사용

data_loc = 'https://github.com/dknife/ML/raw/main/data/'
lifr = pd.read_csv(data_loc + 'lifr_expectancy.csv')
life.head()

#원하는 열만 지정하여 보기
life = life[['Year', 'Alchol', 'Polio', 'BMI', 'GDP']]
print(life)

print(life.shape)
print(life.isnull().sum())
 #결손값 지우기
life.dropna(inplace = True)
print(life.shape)

#상관행렬형성
sns.set(rc= {'figure.figsize':(12, 10)}) #상관행렬 가시
correlation_matrix = life.corr().round(2) #상관행렬 생성
sns.heatmap(data = correlation_matrix, annot=True)

#쌍그림으로 상관관계 파악
sns.pairplot(life[['Life exepectancy', 'Alchol', 'Measles', 'Poilo', 'BMI', 'GDP']])

#학습 완료시 회귀 모델을 얻음
#한번도 본 적이 없는 새로운 데이터로 테스트
#입력데이터는 x에 정답은 y에 훈련용 x를 x_train에 학습결과의 성능을 테스트하기 위해 나머지는 x_test에
#train_test_split으로 데잍를 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, x, test_size= 0.2)

from sklearn.linear_model import LinearRegression
lin_model = LinearRegression() #학습을 진행
lin_model.fit(X_train, y_train)

#기울기가 1에 가까울수록 예측과 정답이 일치
y_hat_train = lin_model.predict(X_train)
plt.scatter(y_train, y_hat_train)
xy_range = [40, 100]
plt.plot(xy_range, xy_range)

#훈련용이 아닌 검증용 데이터를 사용하기
y_hat_train = lin_model.predict(X_test)
plt.scatter(y_test, y_hat_test)
plt.plot(xy_range, xy_range)

#선형회귀 사용시 원래 데이터를 그대로 사용하여 특징값으로 사용 but 범위가 다른 문제가 생김
#각각의 특징들이 갖는 값들을 적당한 규모로 변경하는 작업이 필요 = 정규화

from sklearn.preprocessing import normalize

n_X = normalize(X, axis = 0) #정규화를 0번 축으로 실시
nX_train, nX_test, y_train, y_test = train_test_split(n_X, y, test_size= 0.2)
lin_model.fit(nX_train, y_train)

#선형회귀 모델을 정답과 비교
y_hat_train = lin_model.predict(nXtrain)
y_hat_test = lin_model.predict(nXtest)
plt.scatter(y_train, y_hat_train, color = 'r')
plt.scatter(y_test, y_hat_test, color = 'b')
plt.plot(xy_range, xy_range)

