from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def model_fit(classifier, X, y, save=False, random_state=0):
    if classifier == 'random_forest':
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1, n_estimators=50)
    elif classifier == 'logistic_regression':  
        model = LogisticRegression(random_state=random_state, solver='lbfgs', multi_class='multinomial')
    else:
        return print('You must specify a valid model (random_forest, logistic_regression)')
        
    model.fit(X, y)

    if save:
        fname = '_'.join(classifier, time.strftime("%Y%m%d-%H%M%S"), '.pickle')
        with open(fname, "wb") as f:
            pickle.dump(model, open(fname, 'wb'))
    return model 
