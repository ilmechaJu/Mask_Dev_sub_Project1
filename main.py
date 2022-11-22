import uvicorn
from fastapi import FastAPI

#ML pkg
import joblib, os 

#Vectorizer
gender_vectorizer = open("models/gender_vectorizer.pkl", "rb")
gender_cv = joblib.load(gender_vectorizer)

#Models
gender_nv_model = open("models/gender_nv_model.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)

#ML Aspect
@app.get('/predict/{name}')
async def predict(name):
    vectorized_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorized_name)
    if prediction[0] == 0:
        result = "female"
    else:
        result = "male"

    return {"orig_name":name, "prediction":result}


app = FastAPI()

@app.get('/')
async def index():
    return {"text":"Hello API Builders"}

@app.get('/items/{name}')
async def get_items(name):
    return {"name":name}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)