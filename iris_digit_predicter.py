import streamlit as st
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler


with open('iris_steps.pkl', 'rb') as file:
    data_iris = pickle.load(file)
with open('digits_steps.pkl', 'rb') as file:
    data_digits = pickle.load(file)

# Function to make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)


def main():
    st.title("IRIS and MNIST predicter")
    dataset = st.sidebar.selectbox("Select Dataset", ("IRIS", "Digits"))
    model = st.sidebar.selectbox("Select Model", ("Logistic Regression", "Neural Network", "Naive Bayes"))
    if(dataset == "IRIS"):    
        if model == "Logistic Regression":
            selected_model = data_iris["log_reg"]
        elif model == "Neural Network":
            selected_model = data_iris["mlp"]
        else:
            selected_model = data_iris["nb"]
    else:
        if model == "Logistic Regression":
            selected_model = data_digits["log_reg"]
        elif model == "Neural Network":
            selected_model = data_digits["mlp"]
        else:
            selected_model = data_digits["nb"] 
    st.write(f"Current Model: {model}")

    if dataset == "IRIS":
        st.write("Use the slider to adjust the input features of the flower:")
        sepal_length = st.slider("Sepal Length", min_value=4.0,max_value=8.0,step=0.1)
        sepal_width = st.slider("Sepal Width", min_value=2.0,max_value=5.0,step=0.1)
        petal_length = st.slider("Petal Length", min_value=0.0,max_value=7.0,step=0.1)
        petal_width = st.slider("Petal Width", min_value=0.0,max_value=3.0,step=0.1)
        
        user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    else:
        st.write("Please upload a digit image:")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.resize((8, 8))
            image_array = np.array(image.convert('L'))
            st.image(image_array, caption='Uploaded Image', use_column_width=True)
            user_input = [image_array.flatten()]

    if st.button("Predict"):
        scaler = StandardScaler()
        user_input = scaler.fit_transform(user_input)
        predictions = make_predictions(selected_model, user_input)
        if(dataset =="IRIS"):
            if(predictions == 0):
                st.write("Predictions:", "Setosa")
            elif(predictions == 1):
                st.write("Predictions:", "Versicolor")
            elif(predictions == 2):
                st.write("Predictions:", "Virginica")
        else:
            st.write("Predictions:", predictions)

if __name__ == "__main__":
    main()
