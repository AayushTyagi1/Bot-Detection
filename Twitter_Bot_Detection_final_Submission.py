# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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
classifier=SVC(kernel='rbf', random_state=0)
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


#Best Method - Decision Tree
from sklearn.tree import DecisionTreeClassifier as DTC
classifier= DTC(criterion="entropy")
classifier.fit(X,y)
test_data = pd.read_csv("test_data_4_students.csv");
test_data.drop(['id_str', 'screen_name', 'location', 'description', 'url', 'created_at', 'lang', 'status', 'has_extended_profile','name'],axis=1,inplace=True)


X1 = training_data.iloc[:, :-1].values

from sklearn.preprocessing import LabelEncoder
Labelx=LabelEncoder()
X1[:,6]=Labelx.fit_transform(X1[:,6])
X1[:,7]=Labelx.fit_transform(X1[:,7])
X1[:,8]=Labelx.fit_transform(X1[:,8])
y1_pred = classifier.predict(X1)
y1_pred=pd.DataFrame(y1_pred);
y1_pred.to_csv("Result.csv",index=False);


My_Own_algorithm

class twitter_bot(object):
    def __init__(self):
        pass

    def perform_train_test_split(df):
        msk = np.random.rand(len(df)) < 0.75
        train, test = df[msk], df[~msk]
        X_train, y_train = train, train.iloc[:,-1]
        X_test, y_test = test, test.iloc[:, -1]
        return (X_train, y_train, X_test, y_test)

    def bot_prediction_algorithm(df):
        train_df = df.copy()
        # converting id to int
        train_df['id'] = train_df.id.apply(lambda x: int(x))
        bag_of_words_bot = r'BOT|Bot|bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                           r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                           r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                           r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'


        # converting verified into vectors
        train_df['verified'] = train_df.verified.apply(lambda x: 1 if ((x == True) or x == 'TRUE') else 0)
        # check if the name contains bot or screenname contains b0t
        condition = ((train_df.name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.description.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.screen_name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.status.str.contains(bag_of_words_bot, case=False, na=False))
                     )  # these all are bots
        predicted_df = train_df[condition]  # these all are bots
        predicted_df.bot = 1
        predicted_df = predicted_df[['id', 'bot']]

        # check if the user is verified
        verified_df = train_df[~condition]
        condition = (verified_df.verified == 1)  # these all are nonbots
        predicted_df1 = verified_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        # check if description contains buzzfeed
        buzzfeed_df = verified_df[~condition]
        condition = (buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False))  # these all are nonbots
        predicted_df1 = buzzfeed_df[buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False)][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        # check if listed_count>16000
        listed_count_df = buzzfeed_df[~condition]
        listed_count_df.listed_count = listed_count_df.listed_count.apply(lambda x: 0 if x == 'None' else x)
        listed_count_df.listed_count = listed_count_df.listed_count.apply(lambda x: int(x))
        condition = (listed_count_df.listed_count > 16000)  # these all are nonbots
        predicted_df1 = listed_count_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])
        #remaining
        predicted_df1 = listed_count_df[~condition][['id', 'bot']]
        predicted_df1.bot = 0 # these all are nonbots
        predicted_df = pd.concat([predicted_df, predicted_df1])
        return predicted_df

    def get_predicted_and_true_values(features, target):
        y_pred, y_true = twitter_bot.bot_prediction_algorithm(features).bot.tolist(), target.tolist()
        return (y_pred, y_true)

    def get_accuracy_score(df):
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        # predictions on training data
        y_pred_train, y_true_train = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        Performance(y_pred_train, y_true_train)

if __name__ == '__main__':
    train_ = pd.read_csv('training_data_2_csv_UTF.csv')
    test_ = pd.read_csv('test_data_4_students.csv')
    

    train_.drop(['id_str', 'location', 'url', 'created_at', 'lang', 'has_extended_profile'],axis=1,inplace=True)
    test_.drop(['id_str', 'location', 'url', 'created_at', 'lang', 'has_extended_profile'],axis=1,inplace=True)  
    predicted_df = twitter_bot.bot_prediction_algorithm(test_)   
    twitter_bot.get_accuracy_score(train_)
