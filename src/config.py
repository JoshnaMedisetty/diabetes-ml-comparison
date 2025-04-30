
import os

RANDOM_STATE = 42
DATA_PATH = 'data/diabetes.csv'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

os.makedirs(RESULTS_DIR, exist_ok=True)

