import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from flask import Flask
from flask import request
from flask import jsonify

app=Flask('churn')

# Load the model and dv
# model_file = "model1.bin" # Uncomment this line if the model used is the model1
model_file = "model2.bin" # Uncomment this line if the model used is the model2

dv_file = "dv.bin"
with open(model_file, 'rb') as model:
    model = pickle.load(model)
with open(dv_file, 'rb') as dv:
    dv = pickle.load(dv)

@app.route('/predict', methods=['POST'])
def predict():
    customer= request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    
    return jsonify(result)
    
if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)