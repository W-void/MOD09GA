#%%
import pandas as pd
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, \
    GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.externals import joblib


#%%
df = pd.read_csv('./csv/229.csv')
col = df.columns
X, Y = df[col[:-1]], df[col[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                        test_size=0.33, stratify=Y)

ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_depth=3,
                                       random_state=0),
    "RF": RandomForestClassifier(n_estimators=10, max_depth=3),
    # "Linear regression": LinearRegression(),
    # "Ridge": RidgeCV(),
    # "SVM": svm.SVC(kernel='poly'), # 太慢了
    "DT": tree.DecisionTreeClassifier(max_depth=3),
    "GBDT": GradientBoostingClassifier(n_estimators= 10, max_depth=3)
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)
    print('{0:20}: {1}'.format(name, accuracy_score(y_test, y_test_predict[name]>0.5)))
    if name == "DT":
        dot_data = tree.export_graphviz(estimator, out_file=None, 
                            feature_names=col[:-1],  
                            class_names=['no_cloud', 'cloud'], 
                            filled=True, rounded=True, 
                            special_characters=True)  
        graph = graphviz.Source(dot_data, filename='tree', directory='./log/', format='png')
        graph.view()

joblib.dump(ESTIMATORS, "./pkl/train_model.pkl")         
#%%
