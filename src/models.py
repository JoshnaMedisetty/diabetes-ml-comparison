# diabetes-ml-comparison/src/models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def initialize_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=500, learning_rate=0.1, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'DNN': build_dnn()
    }

def build_dnn(input_shape=(8,)):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

