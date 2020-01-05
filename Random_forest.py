import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import csv

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

data = pd.read_csv("submit_2019.csv",encoding = 'unicode_escape')
X = data[data.columns[:-2]]

cats = ['education','workclass','marital-status','occupation','relationship','race','sex','native-country']
dummies = pd.get_dummies(X[cats])
temp = pd.concat([X,dummies],axis=1)
X = temp.drop(cats,axis=1)

missing_columns = []
for column in X_train.columns:
    if column not in X.columns:
        missing_columns.append(column)

for column in missing_columns:
    data[column] = 0

X = data[data.columns]
dummies = pd.get_dummies(X[['y (yes or no)  ','probability of yes (or score)']])
temp = pd.concat([X,dummies],axis=1)
X = temp.drop(['y (yes or no)  ','probability of yes (or score)'],axis=1)

cats = ['education','workclass','marital-status','occupation','relationship','race','sex','native-country']
dummies = pd.get_dummies(X[cats])
temp = pd.concat([X,dummies],axis=1)
X = temp.drop(cats,axis=1)

Y_pred = RFClf.predict(X)
Y_proba = RFClf.predict_proba(X)
Y_proba_yes = [Y_proba[i][1] for i in range(len(Y_proba))]

in_file = open("submit_2019.csv", "r")
reader = csv.reader(in_file)
out_file = open("submit_2019_result.csv", "w")
writer = csv.writer(out_file)
i = 0
name = True
for row in reader:
    if name == False:
        row[14] = Y_pred[i]
        row[15] = Y_proba_yes[i]
        i+=1
    if name == True:
        name = False
    writer.writerow(row)
in_file.close()    
out_file.close()