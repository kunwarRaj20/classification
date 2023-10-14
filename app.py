import streamlit as st
import pandas as pd
import joblib
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

with st.container():
    st.title("Student Dropout Predictor")
    st.write("Prediction of a student whether drops out from education based on various factors")

with open('app.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

lr = joblib.load('dropout_prediction_model.pkl')
le = joblib.load('encoder.pkl')

input_names = {
                'Marital status':'Marital status',
                'Application mode' : 'Application mode',
                'Application order':'Application order',
                'Course':'Course',
                'Daytime/evening attendance': 'Daytime/evening attendance',
                'Previous qualification': 'Previous qualification',
                'Displaced':'Displaced',
                'Debtor' : 'Debtor', 
                'Tuition fees up to date' : "Tuition fees up to date", 
                'Gender' : "Gender",
                'Scholarship holder' : "Scholarship holder",
                'Age':"Age", 
                'Curricular units 1st sem (enrolled)' : "Curricular units 1st sem (enrolled)", 
                'Curricular units 1st sem (approved)' : "Curricular units 1st sem (approved)",
                'Curricular units 1st sem (grade)' :"Curricular units 1st sem (grade)",
                'Curricular units 2nd sem (enrolled)' :"Curricular units 2nd sem (enrolled)",
                'Curricular units 2nd sem (approved)' : "Curricular units 2nd sem (approved)", 
                'Curricular units 2nd sem (grade)' : "Curricular units 2nd sem (grade)"
            }


input_type = {
                'Marital status': ['','Single','Married','Widower','Divorced','Facto union','Legally Separated'],
                'Application mode' : ['',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], 
                'Application order' : ['',0,1,2,3,4,5,6,7,8,9], 
                'Course':['','Biofuel Production Technologies','Animation and Multimedia Design',' Social Service (evening attendance)','Agronomy','Communication Design','Veterinary Nursing','Informatics Engineering','Equiniculture','Management','Social Service','Tourism','Nursing',
                            'Oral Hygiene','Advertising and Marketing Management','Journalism and Communication','Basic Education','Management (evening attendance)'],
                'Daytime/evening attendance': ['','No','Yes'],
                'Previous qualification': ['','Biofuel Production Technologies','Animation and Multimedia Design',' Social Service (evening attendance)','Agronomy','Communication Design','Veterinary Nursing','Informatics Engineering','Equiniculture','Management','Social Service','Tourism','Nursing',
                    'Oral Hygiene','Advertising and Marketing Management','Journalism and Communication','Basic Education','Management (evening attendance)'],
                'Displaced' : ['','No','Yes'], 
                'Debtor' : ['','No', 'Yes'],
                'Tuition fees up to date':['','No','Yes'],
                'Gender' : ['','Female','Male'],  
                'Scholarship holder' : ['','No','Yes'],
                'Age': ['',17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 70],
                'Curricular units 1st sem (enrolled)' : ['',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 26],
                'Curricular units 1st sem (approved)' : ['',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 26],
                'Curricular units 1st sem (grade)' : ['',0,1,2,3,4],
                'Curricular units 2nd sem (enrolled)' : ['',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 26],
                'Curricular units 2nd sem (approved)' : ['',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 26],
                'Curricular units 2nd sem (grade)' : ['',1,2,3,4]
            }

input_lst=[]


with st.form(key="my_form", clear_on_submit=False):
    selected_features_list = lr.feature_names
    for i in selected_features_list:
        if i=='Age':
            ele = st.slider("Age",15,70)
        elif i=='Previous qualification':
            ele = st.slider("Previous qualification",1,20)
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
        else:
            ele = st.selectbox(input_names[i], input_type[i])
        input_lst.append(ele)
    submitted = st.form_submit_button("Test", type='primary')

if submitted:

    X_test_input_cols = list(lr.feature_names) 
    default_dict = {}
    for i in range(len(X_test_input_cols)):
        default_dict[X_test_input_cols[i]] = input_lst[i]

    X_input_test = pd.DataFrame(default_dict,index=[0])
    for name in X_test_input_cols:
        if name == 'Marital status':
            switch_value = X_input_test[name].map({'Single': 1,'Married': 2,'Widower':3,'Divorced':4,'Facto union':5,'Legally Separated':6 })
            X_input_test[name] = switch_value 
        elif name=='Course':
            switch_value = X_input_test[name].map({ 'Biofuel Production Technologies' : 1,'Animation and Multimedia Design':2,' Social Service (evening attendance)' : 3,'Agronomy':4,'Communication Design' :5,
                                                   'Veterinary Nursing': 6,'Informatics Engineering':7,'Equiniculture':8,'Management':9,'Social Service':10,'Tourism':11,'Nursing':12,
                                                   'Oral Hygiene':13,'Advertising and Marketing Management':14,'Journalism and Communication':15,'Basic Education':16,'Management (evening attendance)':17})
            X_input_test[name] = switch_value 
        elif name=='Daytime/evening attendance': 
            switch_value = X_input_test[name].map({'Yes': 1,'No': 0})
            X_input_test[name] = switch_value 
        elif name=='Displaced': 
            switch_value = X_input_test[name].map({'Yes': 1,'No': 0})
            X_input_test[name] = switch_value 
        elif name=='Debtor' :
            switch_value = X_input_test[name].map({'Yes': 1,'No': 0})
            X_input_test[name] = switch_value 
        elif name=='Tuition fees up to date':
            switch_value = X_input_test[name].map({'Yes': 1,'No': 0})
            X_input_test[name] = switch_value 
        elif name=='Gender':
            switch_value = X_input_test[name].map({'Male': 1,'Female': 0})
            X_input_test[name] = switch_value 
        elif name=='Scholarship holder' :
            switch_value = X_input_test[name].map({'Yes': 1,'No': 0})
            X_input_test[name] = switch_value 
    
    if input_lst.count('') > 0:
        st.error('Some inputs are missing')
    else:
        y_input_pred = lr.predict(X_input_test)
        if y_input_pred[0]==1:
            st.success('The Student will not dropout 😆😆😆')
        else:
            st.error('The Student will dropout 😭😭😭')
    
if st.button("Reset"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")