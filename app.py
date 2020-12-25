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
    

if __name__ == '__main__':
    main()


