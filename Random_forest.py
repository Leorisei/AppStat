import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv("train_test_2019.csv",encoding = 'unicode_escape')
X = data[data.columns[:-1]]
Y = data.y

cats = ['education','workclass','marital-status','occupation','relationship','race','sex','native-country']
dummies = pd.get_dummies(X[cats])
temp = pd.concat([X,dummies],axis=1)
X = temp.drop(cats,axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

RFClf = RandomForestClassifier(n_estimators=100, criterion = "entropy")
RFClf.fit(X_train, Y_train)
Y_pred = RFClf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


lb = preprocessing.LabelBinarizer()
pred = RFClf.predict_proba(X_test)[:,1]
test = lb.fit_transform(Y_test).flatten()

lr_auc = metrics.roc_auc_score(test, pred)
print('ROC AUC=%.3f' % (lr_auc))
lr_fpr, lr_tpr, _ = metrics.roc_curve(test, pred)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

C = metrics.confusion_matrix(Y_test,Y_pred,labels=["no","yes"])
print(C)
P = C[1][1]/(C[1][1]+C[0][1])
R = C[1][1]/(C[1][1]+C[1][0])
print("F_Measure:",2*P*R/(P+R))