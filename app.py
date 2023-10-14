import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from joblib import Parallel, delayed
import joblib

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

with st.container():
    st.title("Student Dropout Predictor")
    st.write("Prediction of a student whether drops out from the education based on various factors")

with open('app.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

model1 = joblib.load('best-model.pkl')
model2 = joblib.load('test-model.pkl')
le = joblib.load('encoder.pkl')
model = joblib.load('model.pkl')
model4 = joblib.load('model_new.pkl')
model3 = joblib.load('test-feat.pkl')

st.write(model4.feature_names)
# st.write(model.feature_names)

input_names = {'school' : 'Select School type', 'gender' : 'Gender', 'age' : "Age", 'address' : "Address",'famsize':"Family Size", 'Pstatus' : "Pstatus", 
               'Medu' : "Mother Education",'Fedu' :"Father Education",'Mjob' : "Mother Job", 'Fjob' : "Father Job",'reason' : "Reason",'guardian' :"Guardian",
               'traveltime' : "Travel Time(hrs)",'studytime' : "Study Time(hrs)",'failures' :"Failures",'schoolsup' : "School Support",'famsup' : "Family Support",'paid' : 
               "Fee paid",'activities' : "Activities",'nursery':"Nursery",'higher' : "Higher Education??",'internet' : "Internet",'romantic' : "Romantic",'famrel' : "Family relatives",
               'freetime' :"Free Time(hrs)",'goout' : "Vacation/Go out Time(hrs)",'Dalc' : "Dalc",'Walc' : "Walc",'health' : "Health",'absences' : "Days absent"}
input_type = {'school' : ['',"MS","GP"], 'gender' :['',"M","F"], 'address' : ['',"R","U"],'famsize':['',"GT3","LE3"], 'Pstatus' : ['',"T","A"], 'Medu' : ['',0,1,2,3,4],'Fedu' : ['',0,1,2,3,4],'Mjob' : ['',"Teacher","at home","health","services","other"],'Fjob' : ['',"Teacher","at home","health","services","other"],'reason' : ['',"Reputation","Course","Home","other"],'guardian' :
               ['',"Father","Mother","Other"],'traveltime' : ['',1,2,3,4],'studytime' : ['',1,2,3,4],'failures' : ['',0,1,2,3],'schoolsup' : ['',"Yes","No"],'famsup' : ['',"Yes","No"],'paid' : ['',"Yes","No"],'activities' : ['',"Yes","No"],'nursery' : ['',"Yes","No"],'higher' : ['',"Yes","No"],'internet' : ['',"Yes","No"],'romantic' : ['',"Yes","No"],'famrel' : ['',1,2,3,4],'freetime' : ['',1,2,3,4],'goout' : ['',1,2,3,4],'Dalc' : ['',1,2,3,4],'Walc' : ['',1,2,3,4],'health' : ['',1,2,3,4]}
input_lst=[]


input_names = {
                'Course':'Course',
                'Application mode' : 'Application mode', 
               'Displaced':'Displaced',
               'Debtor' : 'Debtor', 
               'Tuition fees up to date' : "Tuition fees up to date", 
               'Gender' : "Gender",
               'Scholarship holder' : "Scholarship holder",
               'Curricular units 1st sem (evaluations)':'Curricular units 1st sem (evaluations)',
               "Curricular units 2nd sem (evaluations)": "Curricular units 2nd sem (evaluations)",
               'Age':"Age", 
               'Curricular units 1st sem (enrolled)' : "Curricular units 1st sem (enrolled)", 
               'Curricular units 1st sem (approved)' : "Curricular units 1st sem (approved)",
               'Curricular units 1st sem (grade)' :"Curricular units 1st sem (grade)",
               'Curricular units 2nd sem (enrolled)' :"Curricular units 2nd sem (enrolled)",
               'Curricular units 2nd sem (approved)' : "Curricular units 2nd sem (approved)", 
               'Curricular units 2nd sem (grade)' : "Curricular units 2nd sem (grade)"
            }


input_type = {
                'Application mode' : ['',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22], 
                'Displaced' : ['',0,1], 
                'Debtor' : ['',0, 1],
                'Tuition fees up to date':['',0,1],
                'Gender' : ['',0,1],  
                'Scholarship holder' : ['',0,1],
                'Age': ['',0,1],
                'Curricular units 1st sem (evaluations)' : ['',0,1,2,3,4],
                'Curricular units 2nd sem (evaluations)' : ['',0,1,2,3,4],
                'Curricular units 1st sem (enrolled)' : ['',0,1,2,3,4],
                'Curricular units 1st sem (approved)' : ['',0,1,2,3,4],
                'Curricular units 1st sem (grade)' : ['',0,1,2,3,4],
                'Curricular units 2nd sem (enrolled)' : ['',0,1,2,3,4],
                'Curricular units 2nd sem (approved)' : ['',0,1,2,3,4],
                'Curricular units 2nd sem (grade)' : ['',1,2,3,4]
            }

input_lst=[]


with st.form(key="my_form", clear_on_submit=False):
    selected_features_list = model3
    # model4.feature_names
    for i in selected_features_list:
        if i=='Age':
            ele = st.slider("Age",15,70)
        elif i=='Course':
            ele = st.slider("Course",1,20)
        elif i=='Curricular units 1st sem (enrolled)': 
            ele = st.slider("Curricular units 1st sem (enrolled)",0,30)   
        elif i=='Curricular units 1st sem (approved)': 
            ele = st.slider("Curricular units 1st sem (approved)",0,30)   
        elif i=='Curricular units 2nd sem (enrolled)': 
            ele = st.slider("Curricular units 2nd sem (enrolled)",0,30)   
        elif i=='Curricular units 2nd sem (approved)' :
            ele = st.slider("Curricular units 2nd sem (approved)",0,30)   
        elif i=='Curricular units 1st sem (grade)':
            ele = st.slider("Curricular units 1st sem (grade)",0,30)
        elif i=='Curricular units 2nd sem (grade)' :
                ele = st.slider("Curricular units 2nd sem (grade)",0,30) 
        elif i=='Curricular units 1st sem (evaluations)' :
            ele = st.slider("Curricular units 1st sem (evaluations)",0,30) 
        elif i=='Curricular units 2nd sem (evaluations)' :
            ele = st.slider("Curricular units 2nd sem (evaluations)",0,30) 
        else:
            ele = st.selectbox(input_names[i], input_type[i])
        input_lst.append(ele)
    submitted = st.form_submit_button("Test")

# reload_btn = st.button('Test another')
if submitted:

    X_test_input_cols = list(model3) 
    # model4.feature_names
    default_dict = {}
    for i in range(len(X_test_input_cols)):
        default_dict[X_test_input_cols[i]] = input_lst[i]
        # st.write(input_lst[i])

    X_input_test = pd.DataFrame(default_dict,index=[0])
    st.write(X_input_test)
    for name in X_test_input_cols:
        if X_input_test[name].dtype =='object':
            st.write(X_input_test[name]) 
            # = le.fit_transform(X_input_test[name])

    y_input_pred = model2.predict(X_input_test)
    # st.write(y_input_pred)
    if input_lst.count('')>0:
        st.error('Some inputs are missing')
    elif y_input_pred[0]==1:
        st.success('The Student will not dropout 😆😆😆')
    else:
        st.error('The Student will dropout 😭😭😭')
    
    # if reload_btn:
    #     st.experimental_rerun()


st.write('')
st.write('')
st.write("The source code can be found here 👉[🔗](https://www.github.com/UndavalliJagadeesh/Student_Dropout_Prediction)")