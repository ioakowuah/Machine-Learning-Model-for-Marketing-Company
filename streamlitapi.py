import pickle 
import streamlit as st

model = pickle.load(open('C:/Users/HP/Desktop/ML Project/RandomForestClassifier_model.pkl','rb'))

def main():
    st.title('Client will subscribe to a term deposit')

    #input Variables
    Age = st.numeric_input('age')
    Job = st.text_input('job')
    Marital = st.text_input('marital')
    Education = st.text_input('education')
    Default = st.text_input('default')
    Balance = st.numeric_input('balance')
    Housing = st.text_input('housing')
    Loan = st.text_input('loan')
    Contact = st.text_input('contact')
    Day = st.numeric_input('day')
    Month = st.text_input('month')
    Duration = st.numeric_input('duration')
    Campaign = st.numeric_input('campaign')
    Pdays = st.numeric_input('pdays')
    Previous = st.numeric_input('previous')
    Poutcome = st.text_input('poutcome')

    #prediction code
    if st.button('Predict'):
        makeprediction = model.predict([[Age,Job,Marital,Education,Default,Balance,Housing,Loan,Contact,Day,Month,Duration,Campaign,Pdays,Previous,Poutcome]])
        output = makeprediction

    if __name__=='__main__':
        main()
