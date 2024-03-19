import streamlit as st
import pandas as pd
import joblib
# Function to load the model
@st.cache_data
def load_model():
    with open('spam_mail_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

@st.cache_data
def load_vectorizer():
    with open('vectorizer', 'rb') as file:
        loaded_vectorizer = joblib.load(file)
    return loaded_vectorizer

# Load your model
loaded_model = load_model()
load_vectorizer = load_vectorizer()


# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Spam Mail Prediction Web App')


    # User inputs: textbox
    # intilaize the user inputs
    message = st.text_area('Enter the mail:', key='mail', value= "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's" ,height=10)
    

    user_inputs = {
        'Message': message
    }
    
    if st.button('Predict'):
        # prediction = loaded_model.predict(pd.DataFrame(user_inputs, index=[0]))
        df = pd.DataFrame(user_inputs, index=[0])
        df_vectorized = load_vectorizer.transform(df['Message'])
        prediction = loaded_model.predict(df_vectorized)
        if prediction[0] == 1:
            st.error('The mail is spam.')
        else:
            st.success('The mail is not spam.')
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            st.json(loaded_model.get_params())
            st.write('Model used: Logistic Regression')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'spam_mail_prediction.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="spam_mail_prediction.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'mail_data.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="mail_data.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Spam-Mail-Prediction)')
    st.write('--'*50)
    
elif options == 'About':
    st.title('About')
    st.write('This web app is developed to predict whether a mail is spam or not. The model is built using the Logistic Regression algorithm.')
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Spam-Mail-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
