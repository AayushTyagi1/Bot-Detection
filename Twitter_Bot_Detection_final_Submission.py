# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random

def Performance(actual_value , predicted_value):
    accuracy = accuracy_score(actual_value , predicted_value) * 100
    precision = precision_score(actual_value , predicted_value) * 100
    recall = recall_score(actual_value , predicted_value) * 100
    f1 = f1_score(actual_value , predicted_value, average='weighted')
    print('Accuracy is {:.4f}%\n Precision is {:.4f}%\n Recall is {:.4f}%\nF1 Score is {:.4f}\n'.format(accuracy, precision, recall, f1))

# Set Data
training_data = pd.read_csv("training_data_2_csv_UTF.csv")
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]

plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.title('Bots Friends vs Followers')
sns.regplot(bots.friends_count, bots.followers_count, color='red', label='Bots')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()

plt.subplot(2,1,2)
plt.title('NonBots Friends vs Followers')
sns.regplot(Nbots.friends_count, Nbots.followers_count, color='blue', label='NonBots')
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

features = ['id', 'id_str', 'screen_name', 'location', 'description', 'url',
       'followers_count', 'friends_count', 'listed_count', 'created_at',
       'favourites_count', 'verified', 'statuses_count', 'lang', 'status',
       'default_profile', 'default_profile_image', 'has_extended_profile',
       'name', 'bot']

mask = np.zeros_like(training_data[features].corr(), dtype = np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(training_data[features].corr(),linewidths = 0.25,vmax = 0.7,square = True,cmap = "BuGn", 
            linecolor = 'w',annot = True,annot_kws = {"size":8},mask = mask,cbar_kws = {"shrink": 0.9})




training_data.drop(['id_str', 'screen_name', 'location', 'description', 'url', 'created_at', 'lang', 'status', 'has_extended_profile','name'],axis=1,inplace=True)


X = training_data.iloc[:, :-1].values
y = training_data.iloc[:, 9].values

from sklearn.preprocessing import LabelEncoder
Labelx=LabelEncoder()
X[:,5]=Labelx.fit_transform(X[:,5])
X[:,7]=Labelx.fit_transform(X[:,7])
X[:,8]=Labelx.fit_transform(X[:,8])


#fitting
from sklearn.ensemble import RandomForestClassifier as rf
classifier= rf(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]

B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)
 
#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)


from sklearn.tree import DecisionTreeClassifier as DTC
classifier= DTC(criterion="entropy")
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)


from sklearn.naive_bayes import GaussianNB as GNB
classifier=GNB()
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)



from sklearn.neighbors import KNeighborsClassifier as knn
classifier=knn(n_neighbors=5)
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)



from sklearn.svm import SVC
classifier=SVC(kernel='linear', random_state=0)
classifier.fit(X,y)
bots = training_data[training_data.bot==1]
Nbots = training_data[training_data.bot==0]
 
B = bots.iloc[:,:-1]
B_y = bots.iloc[:,9]
B_pred = classifier.predict(B)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(B_y,B_pred)
Performance(B_y,B_pred)

NB = Nbots.iloc[:,:-1]
NB_y = Nbots.iloc[:,9]
NB_pred = classifier.predict(NB)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(NB_y,NB_pred)
Performance(NB_y,NB_pred)
