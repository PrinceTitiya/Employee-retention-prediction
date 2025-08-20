import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


csv_path = "HR_dataset2.csv"
model_path = "employee_churn_model.pkl"

def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(),[
                'satisfaction_level',
                'last_evaluation',
                'number_project',
                'average_montly_hours',
                'time_spend_company',
                'Work_accident',
            'promotion_last_5years'
            ]),
            ('nominal',OneHotEncoder(),['departments']),
            ('ordinal',OrdinalEncoder(),['salary'])
        ],
        remainder='passthrough'
    )

    model = RandomForestClassifier()
    pipeline= Pipeline([
        ('preprocessor',preprocessor),
        ('model',model),
    ])
    return pipeline

def main():
    print("loading data from:",csv_path)
    data = pd.read_csv(csv_path)

    if 'Departments ' in data.columns and 'departments' not in data.columns:
        data.rename(columns={"Departments ":"departments"},inplace = True)

    data = data.drop_duplicates()

    X = data.drop(columns=['left'])
    y = data['left']

    #Train / Test split 
    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size = 0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train,y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred) 
    print(f"RandomForest accuracy: {accuracy:.3f}")
    print(f"RandomForest precision: {precision:.3f}")
    print(f"RandomForest recall: {recall:.3f}")

    joblib.dump(pipeline,model_path)
    print(f"saved train pipeline to: {model_path}")

if __name__ == "__main__":
    main()