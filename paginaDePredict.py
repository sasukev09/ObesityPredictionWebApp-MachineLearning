import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Define function to load models from pickle files
def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

# Load models
logistic_model = load_model("LogisticRegression.pkl")
random_forest_model = load_model("RandomForestClassifier.pkl")

# Define function to take input for prediction
def take_input():
    # Create input fields for user input
    age = st.number_input('Enter age', min_value=0, step=1)
    gender = st.selectbox('Select gender', ['Male', 'Female'])
    height = st.number_input('Enter height in cm', min_value=0.0)
    weight = st.number_input('Enter weight in kg', min_value=0.0)
    bmi = st.number_input('Enter Body Mass Index BMI', min_value=0.0)
    physical_activity_level = st.number_input('Enter physical activity level 1-5', min_value=1, max_value=5, step=1)

    # Convert gender to numeric
    gender_numeric = 1 if gender == 'Male' else 0

    # Return input as numpy array
    return np.array([[age, gender_numeric, height, weight, bmi, physical_activity_level]])

# Define function to make predictions
def make_prediction(model, input_data):
    return model.predict(input_data)


def train_and_plot(df):
    le = LabelEncoder()
    df["Gender"] = df[["Gender"]].apply(le.fit_transform)

    y = df["ObesityCategory"]
    X = df.drop("ObesityCategory", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    import warnings
    warnings.filterwarnings("ignore")

    model_list = [RandomForestClassifier(), LogisticRegression()]

    model_name_list = []
    model_accuracies = []

    for i in model_list:
      model = i.fit(X_train_scaled, y_train)
      model_name = model.__class__.__name__
      y_pred = model.predict(X_test_scaled)
      accuracy = accuracy_score(y_test, y_pred)

    model_name_list.append(model_name)
    model_accuracies.append(accuracy)


    # Plotting accuracy graphs for both Models after
    fig, ax=plt.subplots(figsize=(2,2))
    cols = ["magenta" if i < (max(model_accuracies)) else "yellow" for i in model_accuracies]
    sns.barplot(x=model_name_list, y=model_accuracies, ax=ax, palette=cols)
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.axhline(1, lw=1, ls="--", color="red")
    ax.set_xticklabels(["RandomForest", "Logistic"])
    plt.title("Accuracy Results: Logistic Regression", fontsize=18, color="b")
    st.pyplot(fig)

    rf = RandomForestClassifier().fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    plt.figure(figsize=(2,2))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, linewidths=0.6, cmap="copper", fmt='.2g')
    plt.title("Accuracy Results: RandomForest Confusion Matrix", fontsize=18)
    st.pyplot(plt)

  #Plotting pie chart to see where user stands compared to the rest of dataset
def pie_chart(df):
    plt.figure(figsize=(10,8))
    j = 1
    for i in ["ObesityCategory"]:
      plt.subplot(2,2, j+1)
      colors = ['skyblue', 'salmon', 'lightgreen', 'orange']
      sns.barplot(x=df[i].value_counts().index, y=df[i].value_counts(), palette=colors)
      plt.xlabel("")
      plt.subplot(2,2, j)
      colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightpink']
      plt.pie(x=df[i].value_counts(), labels=df[i].value_counts().index, autopct='%.1f%%',
            pctdistance=0.8, shadow=True, colors=colors)
    j += 2
    plt.tight_layout()
    st.pyplot(plt)

def main():
    # Web app titles and sub titles
    st.title('Fitness/Obesity Level Prediction Form :health_worker:')
    st.header("Upload the test data prior execution.")
    st.subheader("CSV File Uploader")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Take input from user
    input_data = take_input()

    # Make predictions using the loaded models
    logistic_prediction = make_prediction(logistic_model, input_data)
    random_forest_prediction = make_prediction(random_forest_model, input_data)

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the DataFrame
        st.write("Uploaded DataFrame:")
        st.write(df)
    
      # Button to display prediction results
    if st.button("Prediction Results"):
            # Display predictions
            st.subheader('Predictions:')
            st.write(f"Logistic Regression prediction: {logistic_prediction}")

   # Button to display accuracy
    if st.button("Show Accuracy of Results"):
           with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_and_plot(df)

 # Button to display pie chart
    if st.button("Your category in relation to the rest of users:"):
           with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pie_chart(df)

if __name__ == "__main__":
    main()