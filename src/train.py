
from sklearn.model_selection import GridSearchCV

def train_model(model, X_train, y_train):
    if isinstance(model, GridSearchCV):
        model.fit(X_train, y_train)
        return model.best_estimator_
    model.fit(X_train, y_train)
    return model

def train_dnn(model, X_train, y_train):
    history = model.fit(X_train, y_train, 
                       epochs=100, 
                       batch_size=32,
                       validation_split=0.2,
                       verbose=0)
    return history

