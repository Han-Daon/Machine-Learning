import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster

#닥스훈트의 길이와 높이 데이터
dach_length = [77, 78 , 85, 73, 77, 73, 80]
dackh_height = [25, 28, 29, 30, 21, 22, 17, 35]
#사모예드 길이와 높이 디이처
samo_length = [75, 77, 86, 86, 79, 83, 83, 88]
samo_height = [56, 57, 50, 53, 60, 53, 49, 61]

plt.scatter(dach_length, dackh_height, c='red', label = 'Dachhund')
plt.scatter(samo_length, samo_height, c = 'blue', marker='^', label = 'Samoyed')

plt.xlabel('Length')
plt.ylabel('Height')
plt.title('Dog size')
plt.legend(loc = 'upper left')

plt.show()

#다음 데이터를 가진 강아지는 과연 어떤 종일지 구분해보자
newdata_length = [79]
newdata_height = [35]

plt.scatter(newdata_length, newdata_height, s = 100, marker = 'p', c='green', label = 'new Data')

#닥스훈트 클래스에 더 가까움

#준비된 데이터에 KNN알고리즘 적용
d_data = np.column_stack((dach_length, dackh_height))
d_label = np.zeros(len(d_data)) #닥스훈트를 0으로 레이블링

s_data = np.column_stack((samo_length, samo_height))
s_label = np.ones(len(s_data)) #사모예드를 1로 레이블링
newdata = [[79, 35]]
dog_classes = {0:'Daschound', 1:'Samoyed'}

k = 3 #k를 3으로 두고 knn분류기 만들기
#k를 5로 두면 samoted가 나온다
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(dofs, labels) #딕셔너리는 fit함수를 통해 생성한 모데링 잘 적용되는가를 보기위해 사용
y_pred = knn.predict(newdata)
print('데이터', newdata, '판정결과', dog_classes[y_pred[0]])


##군집화 : 소속집단 정보가 없고 모르는 상태에서 비슷한 집단으로 묶는 비지도 학습 - 데이터에서 의미를 파악하고 기준을 만드는 목적으로 사용

#닥스훈트의 길이와 사모예드의 길이 값을 가진 두 리스트를 연겨하여 길이 정보만 가진 dog_length 다차원 배열 데이터를 만들기

dog_length = np.array(dach_length + samo_length)
dog_height = np.array(dach_height + samo_height)

dog_data = np.column_stack((dog_length, dog_height))

plt.title("Dog data without label")
plt.scatter(dog_length, dog_height)

#dog_data는 어느 것이 사모예드고 닥스훈트인지에 대한 정보는 제공되지 않는다. kmeans를 사용하여 두 개의 그룹으로 나누자

def kmeans_predict_plot(X, k):
    model = cluster.KMeans(n_clusters=k)
    model.fit(X)
    labels = model.predict(X)
    colors = np.array(['red', 'green', 'blue', 'magenta'])
    plt.suptitle('K-Means clustering, k={}'.format(k))
    plt.scatter(X[:, 0], X[:, 1], color=colors[labels])
    
kmeans_predict_plot(dog_data, k=2)    

#kmeans 알고리즘은 원리가 단순하고 직관적이며 성능이 좋은 군집화 알고리즘이다.
#다만 군집의 개수 k값을 지정해야한다.