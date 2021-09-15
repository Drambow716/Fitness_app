import pandas as pd
from time import time
from sklearn import model_selection

#Data Processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder #OrdinalEncoder and OHE
from sklearn.preprocessing import StandardScaler
from sklearn import impute
from sklearn import compose
from sklearn.model_selection import train_test_split #train_test_split
from sklearn import metrics #accuracy score, balanced_accuracy_score, confusion_matrix
import pickle as pk

def training_pipeline(dataframe,  save_model=False):

    df = dataframe.copy()

    #Preprocessing
    #Variable Definition
    all_vars = ["time","sensor","x","y","z","user","moving","timestampbeg","file"]
    
    cat_vars = ['sensor', 'user', 'moving', 'file'  
                     ]
    num_vars = ['time', 'x', 'y', 'z', 'timestampbeg'
                     ]
    
    #Warning : These Preprocessing Pipelines are PLACEHOLDER!
    cat_pipe = Pipeline([
              ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
              ])
    num_pipe = Pipeline(steps=[
              ('imputer', impute.SimpleImputer(strategy='mean'))
              ('scaling', StandardScaler())
              ])
    
    preprocessing = compose.ColumnTransformer([
              ('numerical', num_pipe, num_vars)
              ('categorical', cat_pipe, cat_vars)
              ])  
    
    #ML Models
    
    from sklearn.tree          import DecisionTreeClassifier
    from sklearn.ensemble      import RandomForestClassifier
    from sklearn.ensemble      import ExtraTreesClassifier
    from sklearn.ensemble      import AdaBoostClassifier
    from sklearn.ensemble      import GradientBoostingClassifier
    from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
    from sklearn.ensemble      import HistGradientBoostingClassifier
    from xgboost               import XGBClassifier
    from lightgbm              import LGBMClassifier
    from catboost              import CatBoostClassifier

    tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Extra Trees":   ExtraTreesClassifier(n_estimators=100),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "AdaBoost":      AdaBoostClassifier(n_estimators=100),
    "Skl GBM":       GradientBoostingClassifier(n_estimators=100),
    "Skl HistGBM":   HistGradientBoostingClassifier(max_iter=100),
    "XGBoost":       XGBClassifier(n_estimators=100, use_label_encoder=False),
    "LightGBM":      LGBMClassifier(n_estimators=100),
    "CatBoost":      CatBoostClassifier(n_estimators=100, verbose=False)
    }

    # X, Y, and Train Test Splitting
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, stratify=y)

    # MODEL BEAUTY CONTEST
    results = pd.DataFrame({'Model': [], 'MSE': [], 'MAPE': [], 'Time': []})

    for model_name, model in tree_classifiers.items():

        start_time = time()
        #FULL PIPELINE DEFINITION
        pipe = Pipeline([
               ('preprocessing', preprocessing),
               ('classifier', model)
               ])
        
        #Model Training
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        
        results = results.append({
                              "Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, y_pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, y_pred)*100,
                              "Time":     total_time
                              },
                              ignore_index=True) 
        
        #Result Sorting
        results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
        results_ord.index += 1 
        results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

        #Determine the best model
        best_model = tree_classifiers[results_ord.iloc[0].Model]
        best_model_name = results_ord.iloc[0].Model
        best_model_accuracy = results_ord.iloc[0].Accuracy
        best_model.fit(x_train, y_train)

        #Saving the Model
        if save_model:
              model_directory = './model/optimal_model.pkl'
              with open(model_directory, 'wb') as file:
                     pk.dump(best_model, file)

    return best_model_name, best_model_accuracy

def testing_pipeline(dataframe,  save_model=False):

    df = dataframe.copy()

    #Preprocessing
    #Variable Definition
    all_vars = ["time","sensor","x","y","z","user","moving","timestampbeg","file"]
    
    cat_vars = ['sensor', 'user', 'moving', 'file'  
                     ]
    num_vars = ['time', 'x', 'y', 'z', 'timestampbeg'
                     ]
    
    #Warning : These Preprocessing Pipelines are PLACEHOLDER!
    cat_pipe = Pipeline([
              ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
              ])
    num_pipe = Pipeline(steps=[
              ('imputer', impute.SimpleImputer(strategy='mean'))
              ('scaling', StandardScaler())
              ])
    
    preprocessing = compose.ColumnTransformer([
              ('numerical', num_pipe, num_vars)
              ('categorical', cat_pipe, cat_vars)
              ])  
    
    #Load Model
    model = pk.load(open('./model/optimal_model.pkl','rb'))

    #Full Pipeline Definition
    pipe = Pipeline([
           ('preprocessing', preprocessing),
           ('classifier', model)
           ])
    
    # X, Y, and Train Test Splitting
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return pipe.predict(x)

    

