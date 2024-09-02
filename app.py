import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app
st.title("Simple Regression Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Load the file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of the data")
    st.write(df.head())

    # Select columns for regression
    columns = df.columns.tolist()
    x_column = st.selectbox("Select the feature column (X)", columns)
    y_column = st.selectbox("Select the target column (Y)", columns)

    if x_column and y_column:
        # Prepare the data
        X = df[[x_column]]
        y = df[y_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display results
        st.write("### Regression Results")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R^2 Score: {r2}")

        st.write("### Model Coefficients")
        st.write(f"Intercept: {model.intercept_}")
        st.write(f"Coefficient for {x_column}: {model.coef_[0]}")

        # Optionally, display a plot
        st.write("### Plot of Actual vs Predicted")
        st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).reset_index(drop=True))

