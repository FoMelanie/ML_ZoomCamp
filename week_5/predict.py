import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# Load the model and dv

model_file = "model1.bin"
dv_file = "dv.bin"

with open(model_file, 'rb') as model:
    model = pickle.load(model)

with open(dv_file, 'rb') as dv:
    dv = pickle.load(dv)
    
# Customer to test
customer = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform(customer)
print("Associated probability for the customer:")
print(model.predict_proba(X)[0,1])