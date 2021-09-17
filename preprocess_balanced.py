# imports
# %%
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost               import XGBClassifier
from sklearn.ensemble      import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix

data = pd.read_csv("dataset_halfSecondWindow.csv")
data.drop("activityrecognition#0", inplace = True, axis = 1)
data.drop("Unnamed: 0", inplace = True, axis = 1)

# Imputing
imputer = SimpleImputer(strategy='constant',fill_value=0)
cols = data.columns
df = pd.DataFrame(imputer.fit_transform(data),columns = cols)

ns_cols =['android.sensor.gravity#mean',
 'android.sensor.gravity#min',
 'android.sensor.gravity#max',
 'android.sensor.gravity#std',
 'android.sensor.light#mean',
 'android.sensor.light#min',
 'android.sensor.light#max',
 'android.sensor.light#std',
 'android.sensor.magnetic_field#mean',
 'android.sensor.magnetic_field#min',
 'android.sensor.magnetic_field#max',
 'android.sensor.magnetic_field#std',
 'android.sensor.magnetic_field_uncalibrated#mean',
 'android.sensor.magnetic_field_uncalibrated#min',
 'android.sensor.magnetic_field_uncalibrated#max',
 'android.sensor.magnetic_field_uncalibrated#std',
 'android.sensor.proximity#mean',
 'android.sensor.proximity#min',
 'android.sensor.proximity#max',
 'android.sensor.proximity#std']
df.drop(ns_cols,axis =1, inplace =True)

users_train = ['U9','U1','U13','U8','U2',
 'U6',
 'U3','U10','U11','U7','U12']
users_test = ['U5','U6']

df_train = df[df.user.isin(users_train)]
df_test = df[df.user.isin(users_test)]

def pairing(data,seq_len =65):
    x = []
    y = []

    for user in data.user.unique():
        for target in data[ (data.user == user )].target.unique():
            
            data_f = data[ (data.user == user) & (data.target == target) ].iloc[:,:-2]
            if len(data_f.id) > seq_len:
                for i in range(0,(len(data_f.id)-seq_len),seq_len):
                    seq = np.zeros( (seq_len,data_f.shape[1]) )
                    for j in range(seq_len):
                        seq[j] =  data_f.iloc[i:i+seq_len,:].values[j]
                    y.append(target)
                    x.append(seq.flatten())

    return np.array(x),np.array(y)

X_train, y_train = pairing(df_train,seq_len = 65)
X_test, y_test = pairing(df_test,seq_len = 65)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def run_learning(model):
    model = model
    model.fit(X_train_scaled,y_train)
    predictions = model.predict(X_test_scaled)
    return accuracy_score(y_test,predictions) , plot_confusion_matrix(model,X_test_scaled,y_test)

accuracy, confusion_matrix = run_learning(RandomForestClassifier())

print(accuracy)

"""
U12   ['Bus' 'Car' 'Still' 'Train' 'Walking']   9081
U3   ['Bus' 'Train' 'Walking']   3522
U6   ['Bus' 'Car' 'Still' 'Train' 'Walking']   3029
U1   ['Bus' 'Car' 'Still' 'Train' 'Walking']   25700
U7   ['Bus' 'Train']   4228
U4   ['Car' 'Still' 'Walking']   2434
U13   ['Car']   957
U10   ['Car' 'Still' 'Walking']   5717
U8   ['Car' 'Still']   2229
U2   ['Car' 'Still' 'Walking']   2897
U11   ['Car']   1780
U5   ['Still' 'Walking']   186
U9   ['Train' 'Walking']   825
"""
