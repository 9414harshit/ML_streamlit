import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Binary Classification web App")
    st.sidebar.title("Binary Classification web App")
    
    #filename = st.text_input('Enter a file path:')
    uploaded_file = st.file_uploader("Choose a file",type=['csv'])
    if uploaded_file is None:
        return
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv(uploaded_file)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data 
    
    df = load_data()

    if st.sidebar.checkbox("show row data", False):
        st.subheader(" Data Set ")
        st.write(df)
    
    t=st.text_input("Write feature name")
    test_size=st.number_input("test size for testing",0.1,0.3,key='ts')
    @st.cache(persist =True)
    def split(df):
        y = df[t]
        x=df.drop(columns=[t])
        x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=0)
        return x_train,x_test,y_train,y_test

    
    df = load_data()
    x_train,x_test,y_train,y_test=split(df)

    def plot_mat(mat):
        if "confusion_Matrix" in mat:
            st.subheader("confuison Matrix")
            plot_confusion_matrix(model,x_test,y_test, display_labels=class_name)
            st.pyplot()

        if 'ROC Curve' in mat:
            st.subheader("ROC curve")
            plot_roc_curve(model,x_test,y_test)
            st.pyplot()
        if 'presion-Recall Curve' in mat:
            st.subheader("Presion recall")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()

    class_name=('ediable','non')
    st.sidebar.subheader("choose classifier")
    classifier = st.sidebar.selectbox("Classifier", ("SVM",'LR','Randomforest',"Linear Regression" ))

    if classifier=='SVM' :
        st.sidebar.subheader("Model hyperparameter")
        C = st.sidebar.number_input("Cds", 0.01, 10.0, key='C')
        kernel= st.sidebar.radio("Kernel",   ("rbf","linear"), key='kernel')
        gamma=st.sidebar.radio("Gamma ", ("scale", "auto"), key="gamma")
        s=st.number_input("ddd",1,1000)
        Matrix = st.sidebar.multiselect("what to plot?",('confusion_Matrix','ROC Curve','presion-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support vector Machine Results")
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("precision: ", precision_score(y_test, y_pred,labels=class_name))
            st.write("Recall: ", recall_score(y_test,y_pred,labels=class_name))
            plot_mat(Matrix)


    if classifier=='LR' :
        st.sidebar.subheader("Model hyperparameter")
        C = st.sidebar.number_input("Cds", 0.01, 10.0, key='LR')
        max_iter = st.sidebar.slider("No. of iteration", 100,500,key='LRi')
        Matrix = st.sidebar.multiselect("what to plot?",('confusion_Matrix','ROC Curve','presion-Recall Curve'))

        
        st.subheader("Support vector Machine Results")
        model=LogisticRegression(C=C,max_iter=max_iter)
        model.fit(x_train,y_train)
        accuracy = model.score(x_test, y_test)
        y_pred=model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("precision: ", precision_score(y_test, y_pred,labels=class_name))
        st.write("Recall: ", recall_score(y_test,y_pred,labels=class_name))
        plot_mat(Matrix)
       
    if classifier=="Linear Regression" :
        st.sidebar.subheader("Model hyperparameter")
        #C = st.sidebar.number_input("C", 0.01, 10.0, key='LRs')
        max_iter = st.sidebar.slider("No. of iteration", 100,500,key='LRi')
        Matrix = st.sidebar.multiselect("what to plot?",('confusion_Matrix','ROC Curve','presion-Recall Curve'))

        
        st.subheader("LR Results")
        model=LinearRegression()
        model.fit(x_train,y_train)
        accuracy = model.score(x_test, y_test)
        y_pred=model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        #st.write("precision: ", precision_score(y_test, y_pred,labels=class_name))
        #st.write("Recall: ", recall_score(y_test,y_pred,labels=class_name))
        #plot_mat(Matrix)

    if classifier=='Randomforest' :
        st.sidebar.subheader("Model hyperparameter")
        C = st.sidebar.number_input("Cds", 100, 1000, key='Cn')
        max_depth= st.sidebar.number_input("max depth", 1,20, key='kel')
        gamma=st.sidebar.radio("bootstrap sample ", ("true", "false"), key="mma")
        Matrix = st.sidebar.multiselect("what to plot?",('confusion_Matrix','ROC Curve','presion-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Randomforest")
            model=RandomForestClassifier(n_estimators=C,max_depth=max_depth,bootstrap=gamma,n_jobs=1)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("precision: ", precision_score(y_test, y_pred,labels=class_name))
            st.write("Recall: ", recall_score(y_test,y_pred,labels=class_name))
            plot_mat(Matrix)
            


if __name__ == '__main__':
    main()


