import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.callback import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#Getting and Splitting the dataset
X,y  =  load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#Trainig XGBoost
'''model = xgb.XGBClassifier()
model.fit(X_train, y_train)

#Making predictions
pred = model.predict(X_test)'''

#Checking accuracy
'''print("The accuracy is, ", accuracy_score(y_test, pred))'''

#Early stopping and evaluation set
'''early_stopping_callback = EarlyStopping(rounds = 10)
model = xgb.XGBClassifier(n_estimators =1000)
model.fit(X_train, y_train, EarlyStopping(round=10), eval_set = [(X_test, y_test)], verbose = True )'''

#Plotting importance
'''xgb.plot_importance(model)
plt.show()'''

#Saving and loading the model
'''x = model.save_model("xgb_saved.json")#Saving the model
l = xgb.XGBClassifier()#Loading the model
o = l.load_model("xgb_saved.json")
print(o)'''

#Using XGBoost with DMatrix for better performance
'''dtrain = xgb.DMatrix(X_train, label  =y_train)
dtest = xgb.DMatrix(X_test, label = y_test)
paras = {
    'max_depth': 3,
    'eta':0.1,
    'objective':'multi:softmax',
    'num_class': 3
}
bst = xgb.train(paras, dtrain, num_boost_round=100, evals = [(dtest, 'test')], early_stopping_rounds=10)'''
'''print("The Best iteration is: ", bst.best_iteration)
print("The Best score is: ", bst.best_score)''' 

#Custom loss function
'''def custom_loss(pred, dtrain):
    labels = dtrain.get_label()
    grad = pred - labels
    hess = grad * (1-pred)
    return grad, hess

#Cross validation with XGBoost
cv_results = xgb.cv(paras, dtrain, num_boost_round = 200, nfold = 5, metrics = 'mlogloss',early_stopping_rounds = 10, as_pandas = True)
print(cv_results.head())'''

#Using GridSearchCV for hyperparameter tuning
'''params = {
    'max_depth': [3,5],
    'learning_rate': [0.01,0.1],
    'n_estimaters': [50,100]
}
grid = GridSearchCV(xgb.XGBClassifier(), params, scoring = 'accuracy', cv = 3)
grid.fit(X_train, y_train)
print(grid.best_params_)'''

#Handling imbalanced data
'''model = xgb.XGBClassifier(scale_pos_weight = len(negative_class) / len(positive_class))'''


