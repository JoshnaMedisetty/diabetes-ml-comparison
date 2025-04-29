
import os

RANDOM_STATE = 42
DATA_PATH = '../data/pima_diabetes.csv'
RESULTS_DIR = '../results'
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

os.makedirs(RESULTS_DIR, exist_ok=True)

