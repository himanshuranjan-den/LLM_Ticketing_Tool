import streamlit as st
from pages.admin_utils import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib


if 'cleaned_data' not in st.session_state:
    st.session_state['cleaned_data'] = ''
if 'sentence_train' not in st.session_state:
    st.session_state['sentence_train'] = ''
if 'sentence_test' not in st.session_state:
    st.session_state['sentence_test'] = ''
if 'labels_train' not in st.session_state:
    st.session_state['labels_train'] = ''
if 'labels_test' not in st.session_state:
    st.session_state['labels_test'] = ''
if 'svm_classifier' not in st.session_state:
    st.session_state['svm_classifier'] = ''



st.title("Build your own classification model")

# Create tabs
tab_titles = ['Data processing', 'Model Training', 'Model Evaluation', 'Save Model']
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.title('Data Processing Zone')
    st.write('Here we process the data')

    data = st.file_uploader('Uplaod your CSV file', type='csv')
    button = st.button('Load data', key='data')

    if button:
        with st.spinner('Loading file...'):
            our_data = read_data(data=data)
            embeddings = get_embeddings()
            st.session_state['cleaned_data'] = create_embeddings(df=our_data, embeddings=embeddings)
        st.success('Data Processing Completed!!')


# Model training tab

with tabs[1]:
    st.title('Model training tab')
    st.write('Here we train the model')
    button = st.button('Train model', key='model')

    if button:
        try:
            with st.spinner('Training model'):
                st.session_state['cleaned_data'].to_csv('test.csv')
                st.session_state['sentence_train'], st.session_state['sentence_test'], st.session_state['labels_train'], st.session_state['labels_test'] = split_train_test_data(st.session_state['cleaned_data'])
                st.session_state['svm_classifier'] = make_pipeline(StandardScaler(), SVC(class_weight='balanced'))
                st.session_state['svm_classifier'].fit(st.session_state['sentence_train'], st.session_state['labels_train'])
            st.success('Model Training Completed!')
        except Exception as e:
            st.write(str(e))
            exit

# Model evaluation tab 

with tabs[2]:
    st.header('Model Evaluation tab')
    st.write('Here we evaluate the model')
    button = st.button('Evaluate mdoel', key='Evaluate')

    if button:
        with st.spinner('Evaluation in progress'):
            score = get_score(svm_classifier=st.session_state['svm_classifier'] ,sentence_test = st.session_state['sentence_test'], label_test = st.session_state['labels_test'])
            st.success('Validation score is {}'.format(score*100))

            st.write('Doing a sample run')
            text = 'Rude driver is driving the bus very badly'
            st.write("***Our Test text *** : {}".format(text))
            embeddings = get_embeddings()
            query_result = embeddings.embed_query(text)
            result = st.session_state['svm_classifier'].predict([query_result])
            st.write("** The query belongs to : {}".format(result[0]))
        st.success('Evaluation Complete!!')


with tabs[3]:
    st.header('Save mdoel')
    st.write('Here we are saving the model')

    button = st.button('Save model', key="Save")

    if button:
        with st.spinner("Saving model"):
            joblib.dump(st.session_state['svm_classifier'], 'modelsvm.pk1')
        st.success("OUr model has been save by the name modelsvm.pk1")