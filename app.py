import streamlit as st
import pandas as pd
import pickle
import numpy as np
import base64
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
st.title('Credit Card Default Classification App')
st.subheader('This app uses some input parameters to predict whether a person will default on his credit card payment.')
data = pd.read_csv('Data/CreditCardDefault_Classification.csv')
st.write('Input Data should be in the form:')
inp = data.drop('def_flag', axis=1)
st.dataframe(inp.sample(1))
test = st.file_uploader('Choose a file from your device:')
if test is not None:
    input_df = pd.read_csv(test)
elif test is None:
    def input_features():
        st.sidebar.title('Looking for a single customer?')
        age = st.sidebar.slider('Age', 18, 100, 30)
        int_rate = st.sidebar.slider('Monthly Interest Rate', float(1.0), float(15.0), float(3), step=0.5)
        bill = st.sidebar.slider('bill Amount', 1000, 1000000, 50000)
        sal = st.sidebar.slider('Monthly Salary', 100, 1000000, 5000)
        contacts = st.sidebar.slider('No. of Contacts', 0, 50000, 1000)
        edu_type = st.sidebar.selectbox('Education Type', [1,2,3], index = 1)
        fb_status = st.sidebar.selectbox('Facebook Status', [0,1], index=1)
        li_status = st.sidebar.selectbox('LinkedIn Status', [0, 1], index=1)
        de_id = st.sidebar.selectbox('Designation ID', [0, 1, 2, 3], index=1)
        loan_purpose = st.sidebar.selectbox('Purpose of Loan', [1,2,3,4,5,6], index=1)
        data = {'age':age,
                'bill_amount': bill,
                'Monthly_int_Rate': int_rate,
                'monthly_salary': sal,
                'education_type_id': edu_type,
                'Purpose_Loan': loan_purpose,
                'facebook_status': fb_status,
                'linkedin_status': li_status,
                'designation_id': de_id,
                'No Of Contats': contacts}
        features = pd.DataFrame(data, index = [0])
        return features
    input_df = input_features()
load_scaler = pickle.load(open('scaler.pkl', 'rb'))
scaled_df = load_scaler.transform(input_df)
load_clf = pickle.load(open('classification_model.pkl', 'rb'))
prediction = load_clf.predict(scaled_df)
prob = load_clf.predict_proba(scaled_df)

if test is None:

    st.subheader('Prediction')
    ans = np.array(['Not likely to default', 'Likely to default'])
    st.write(ans[prediction])

    st.subheader('Prediction Probability')
    st.write(prob)
    default = '**'+str(np.round(prob[0][1], 4))+'**'
    st.write('The customer will default with a probability of ' + default)
else:

    df = pd.DataFrame(data = prediction, columns = ['def_flag'])
    df = pd.concat([input_df,df], axis = 1)
    st.write(df.sample(5))
    towrite = io.BytesIO()
    downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="myfilename.xlsx">Download excel file</a>'
    st.markdown(linko, unsafe_allow_html=True)


