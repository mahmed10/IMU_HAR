import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from time import time

start_time = time()
features_type = ''
test_row = ''
X_train = pd.read_csv('features'+features_type+'_train.csv', header = None)
y_train = pd.read_csv('labels_train.csv', header = None)[0]
print('train read done')


X_train = X_train.values
y_train = y_train.values
train_time = time() - start_time

print(X_train.shape)
print(y_train.shape)

start_time = time()
X_test = pd.read_csv('features'+features_type+'_test'+test_row+'.csv', header = None)
print('test read done')


X_test = X_test.values
print(X_test.shape)
test_time = time()-start_time

#normalization
sc = StandardScaler()
start_time = time()
X_train = sc.fit_transform(X_train)
train_time = train_time + time() - start_time
start_time = time()
X_test = sc.transform(X_test)
test_time = test_time + time() - start_time

print('classifier start')

start_time = time()

classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier = LogisticRegression(random_state = 0)
#classifier = LinearDiscriminantAnalysis()
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
classifier_name = 'RnF 20'
print(classifier.get_params())
train_time = train_time + time() - start_time
#print(classifier)

start_time = time()
# Predicting the Test set results
print('predict')
y_pred = classifier.predict(X_test)
print(y_pred)
test_time = test_time + time() - start_time

print(train_time)
print(test_time)
print(time())
result = np.c_[y_pred]
#np.savetxt(classifier_name+features_type+test_row+'_result.csv', np.array(result), delimiter=",", fmt = '%.1f')