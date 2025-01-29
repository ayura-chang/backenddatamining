from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ayura-chang.github.io/frontenddatamining/","http://localhost:5173", "https://backenddatamining-production.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


data = pd.read_csv('Mental_Health_dataset.csv', delimiter=';')


data = data[['Gender', 'Age', 'Course', 'YearOfStudy', 'CGPA', 'Depression', 'Anxiety',
             'PanicAttack', 'SpecialistTreatment', 'SymptomFrequency_Last7Days', 
             'HasMentalHealthSupport', 'SleepQuality', 'StudyStressLevel', 
             'StudyHoursPerWeek', 'AcademicEngagement']]

data = data.dropna()


label_encoders = {}
for column in ['Gender', 'Course', 'YearOfStudy']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


data['No'] = (data['Depression'] < 1).astype(int)

X = data.drop(columns=['Depression', 'No'])
y = data['No']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
classification_report_str = classification_report(y_test, y_pred)


data['Prediction'] = model.predict(X)


data['Prediction_Label'] = data['Prediction'].map({0: "No Depression", 1: "Depression"})


data['Course'] = label_encoders['Course'].inverse_transform(data['Course'])


data.to_csv('Processed_Mental_Health_dataset.csv', index=False)

@app.get("/accuracy")
def get_accuracy():
    return {"accuracy": f"{accuracy:.2f}%"}

@app.get("/predict")
def get_predictions():
    try:
        processed_data = pd.read_csv('Processed_Mental_Health_dataset.csv')
        predictions = processed_data[['Gender', 'Age', 'Course', 'Depression']].to_dict(orient='records')
        return {"predictions": predictions}
    except FileNotFoundError:
        return {"error": "Processed data file not found. Please train the model first."}

if __name__ == '__main__':
    import uvicorn
    for route in app.routes:
        print(f"Endpoint: {route.path} | Method: {route.methods}")
    uvicorn.run(app, host="127.0.0.1", port=8000)