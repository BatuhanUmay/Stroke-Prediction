import streamlit as st

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data.drop("id", inplace=True, axis=1)

data["bmi"].replace(to_replace=np.NaN, value=data["bmi"].mean(), inplace=True)

le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])
data["ever_married"] = le.fit_transform(data["ever_married"])
data["work_type"] = le.fit_transform(data["work_type"])
data["Residence_type"] = le.fit_transform(data["Residence_type"])
data["smoking_status"] = le.fit_transform(data["smoking_status"])

x = data.iloc[:, :-1].values  # bağımsız değişkenler
y = data.iloc[:, -1:].values  # bağımlı değişkenler

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=100)

sc = StandardScaler()
x_olcekli_train = sc.fit_transform(x_train)
x_olcekli_test = sc.transform(x_test)

y_olcekli_train = sc.fit_transform(y_train)
y_olcekli_test = sc.transform(y_train)

sm = SMOTE(random_state=2)
x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train.ravel())

print('After OverSampling, the shape of train_x: {}'.format(x_train_sm.shape))
print('After OverSampling, the shape of train_y: {}'.format(y_train_sm.shape))

print('After OverSampling, counts of label 1: {}'.format(sum(y_train_sm == 1)))
print('After OverSampling, counts of label 0: {}'.format(sum(y_train_sm == 0)))


st.title("Stroke Prediction")

gender = st.radio('Pick your gender', ['Male', 'Female', 'Other'])

age = st.number_input('Enter your age', value=0, format='%d')


if gender == 'Male':
    _gender = 1
elif gender == 'Female':
    _gender = 0
else:
    _gender = 2


hypertension = st.radio('Do you have hypertension?', ['Yes', 'No'])

if hypertension == 'Yes':
    hyper_tension = 1
else:
    hyper_tension = 0

heartDisease = st.radio('Do you have any heart disease?', ['Yes', 'No'])

if heartDisease == 'Yes':
    heart_disease = 1
else:
    heart_disease = 0

everMarried = st.radio('Have you ever been married?', ['Yes', 'No'])

if everMarried == 'Yes':
    ever_married = 1
else:
    ever_married = 0

workType = st.selectbox('Choose your work type', [
                        'Government Job', 'Private', 'Self-employed', 'Children', 'Never worked'])

if workType == 'Government Job':
    work_type = 0
elif workType == 'Private':
    work_type = 2
elif workType == 'Self-employed':
    work_type = 3
elif workType == 'Never worked':
    work_type = 1
else:
    work_type = 4

residentalArea = st.selectbox(
    'Choose your residental area', ['Rural', 'Urban'])

if residentalArea == 'Rural':
    residental_area = 0
else:
    residental_area = 1

glucoseLevel = st.number_input('Enter your average glucose level', format='%f')

weight = st.number_input('Enter your weight (in kg)', value=1.0)
height = st.number_input('Enter your height (in m)', value=1.0)
# bmi = round(weight / (height**2), 1)
bmi = 25.6
st.write('The bmi is ', bmi)


smokingStatus = st.selectbox('What is your smoking status?', [
                             'Never smoked', 'Smokes', 'Formerly smoked', 'Unknown'])


if smokingStatus == 'Never smoked':
    smoking_status = 2
elif smokingStatus == 'Smokes':
    smoking_status = 3
elif smokingStatus == 'Formerly smoked':
    smoking_status = 1
else:
    smoking_status = 0


predict = st.button('Predict!')

newUser = [_gender, age, hyper_tension, heart_disease, ever_married,
           work_type, residental_area, glucoseLevel, bmi, smoking_status]

# newUser = pd.DataFrame(newUser)


classifier = XGBClassifier(eval_metric='error', learning_rate=0.1)
classifier.fit(x_train_sm, y_train_sm)

# result = newUser.values.reshape((1, -1))
result = np.array(newUser).reshape((1, -1))
y_pred = classifier.predict(result)

# print("y_pred", y_pred)
# print("y_pred shape", y_pred.shape)
# print("y_test", y_test)
# print("y_test shape", y_test.shape)

# y_test = y_test.reshape(y_test.shape[1:])

dogruluk_degeri = accuracy_score(y_test[0], y_pred) * 100

if predict:
    print("Acc Score:", dogruluk_degeri)
    if y_pred == 1:
        print("you're sick")
    else:
        print("you're fine")
