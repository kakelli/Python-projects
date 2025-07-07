import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
#Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

#Training 
'''model = lgb.LGBMClassifier()'''
'''model.fit(X_train, y_train)'''

#Predictions
'''pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))'''

#Dataset handling in LightGBM
train_data = lgb.Dataset(X_train, label = y_train)
test_data = lgb.Dataset(X_train, label = y_train)

#Custom training parameters
'''params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type':'gbdt',
    'num_leaves':31,
    'learning_rate': 0.05,
    'verbose': -1
}
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round = 100)'''

#Categorial features
'''model = lgb.LGBMClassifier()
model.fit(X_train, y_train, categorical_feature=[0,1,2])'''

#GPU Training
'''params = {
    'deivce':'gpu',
    'gpu_platform_id':0,
    'gpu_device_id': 0
}'''

#Custom loss function
'''def custom_loss(y_true, y_pred):
    grad = ...
    hess = ...
    return grad, hess'''

#Feature Importance
'''lgb.plot_importance(model)
plt.show()'''

#Hyperparameter tuning
'''params = {
    'num_leaves':[31,64],
    'learning_rate':[0.01,0.05],
    'n_estimators': [100,200]
}
search = GridSearchCV(lgb.LGBMClassifier(), params, cv = 3, scoring = 'accuracy')
search.fit(X_train,y_train)'''

#Saving and Loading the file
'''x = model.booster_.save_model('model.txt')
print(x)''' #save

#Loading the file
'''bst = lgb.Booster(model_file='model.txt')
print(bst)''' #Load