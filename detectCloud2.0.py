#%%
import numpy as np
import pandas as pd
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, \
    GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#%%
df = pd.read_csv('./csv/1105.csv')
col = df.columns
X, Y = df[col[:-1]], df[col[-1]]
X = X.drop(columns=['B6'])
X['b/r'] = np.clip(X[col[0]] / X[col[2]], 0, 3)
# X[col[3]] = np.where((X[col[3]] < -100) & (X[col[2]] > 0), 10000, X[col[3]])
# X['NDVI'] = np.clip((X[col[3]] - X[col[2]])/np.where(X[col[3]] + X[col[2]]), -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                        test_size=0.33, stratify=Y)

ESTIMATORS = {
    # "Extra trees": ExtraTreesRegressor(n_estimators=10, max_depth=3,
    #                                    random_state=0),
    # "RF": RandomForestClassifier(n_estimators=10, max_depth=3),
    # "Linear regression": LinearRegression(),
    # "LDA": LinearDiscriminantAnalysis(),
    # "Logistic": LogisticRegression(),
    # "SVM": svm.SVC(kernel='linear'), # 太慢了
    "DT": tree.DecisionTreeClassifier(max_depth=3),
    # "GBDT": GradientBoostingClassifier(n_estimators= 10, max_depth=3)
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    x_, y_ = X_train, y_train
    if name == 'svm':
        idx = pd.random.choice(len(y_train), 1000, replace=False)
        x_, y_ = X.iloc[idx], Y.iloc[idx]

    estimator.fit(x_, y_)
    y_test_predict[name] = estimator.predict(X_test)
    print('{0:20}: {1}'.format(name, accuracy_score(y_test, y_test_predict[name]>0.5)))
    
    if name == "DT":
        dot_data = tree.export_graphviz(estimator, out_file=None, 
                            feature_names=X.columns,  
                            class_names=['no_cloud', 'cloud'], 
                            filled=True, rounded=True, 
                            special_characters=True)  
        graph = graphviz.Source(dot_data, filename='tree', directory='./log/', format='png')
        graph.view()

joblib.dump(ESTIMATORS, "./pkl/train_model.pkl")         
#%%
