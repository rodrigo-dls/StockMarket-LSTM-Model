import streamlit as st
import datetime

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
# import matplotlib.pyplot as plt
import plotly.express as px

# Data normalization
from sklearn.preprocessing import MinMaxScaler

# Learning models
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Validation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cloud Database connection
from google.cloud import firestore

# <----------- Mirar -----------> 
#  Este bien fijar la randomibilidad para el uso de la aplicacion?
tf.random.set_seed(4)

# ***************
# Cache
# ***************

# <----------- Mirar -----------> 
# La fc en cache para conectar al db da error, vale la pena hacerlo?

# @st.cache_data
# def connect_db(credential_file):
#     # Authenticate to Firestore DB with the JSON account key.
#     db = firestore.Client.from_service_account_json(credential_file)
#     return db

# <-------- Hasta aca----------->

@st.cache_data
def get_data_with_date_index(filename):
    """
    Import csv data with "Date" column as the Index

    Returns:
        dataFrame
    """
    data = pd.read_csv(filename, index_col="Date", parse_dates=["Date"])
    return data

@st.cache_data
def add_new_training_data(input_feature, n_epochs, n_batch, units_number, metrics_dict, record_df):
    """
    Registers training data into a dataframe.

    Returns:
        dataFrame: dataFrame of records record_df
    """

    # Get the current datetime in string format
    timestamp = datetime.datetime.now()
    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S') # format the date and time into a string

    # <----------- Mirar -----------> 
    # Es necesario crear esta funcion? Donde mas la usaria? Porque quiero cargar las fechas en formato string y no fechas

    # @st.cache_data
    # def get_formatted_timestamp():
    #     """
    #     Get the current datetime in string format.

    #     Returns:
    #         str: Formatted timestamp (YYYY-MM-DD HH:mm:ss).
    #     """
    #     timestamp = datetime.datetime.now()
    #     formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    #     return formatted_timestamp
    
    # current_timestamp = get_formatted_timestamp()

    # <-------- Hasta aca----------->

    # Store selected parameters
    new_inputs = [input_feature, n_epochs, n_batch, units_number]

    # Store new metrics
    new_metrics = []
    for _, meta in metrics_dict.items():
        new_metrics.append(meta['value'])

    # Create new record
    new_row =  [formatted_timestamp] + new_inputs + new_metrics

    # Add new row to record_df
    record_df.loc[len(record_df.index)] = new_row

    return record_df

@st.cache_data
def create_model(input_dim, output_dim, na):
    """
    Create LSTM Recurrent Neural Network

    Returns:
        model: compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=na, input_shape=input_dim))    # LSTM Layer
    model.add(Dense(units=output_dim))  # Output Layer (without activation function)
    model.compile(optimizer='rmsprop', loss='mse')  # Compilation
    return model

# ***************
# App Configuration & Structure
# ***************

st.set_page_config(
    # Page configuration
    page_title="Stock Market Predictor App",
    page_icon="📈",
    layout="wide"
)

# App Sections
top_test = st.container()
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
metrics = st.container()
viz = st.container()
training_record = st.container()

# ***************
# Header
# ***************

with header:
    st.title("Interactive Stock Price Predictor: Customize Your Forecasts")
    st.markdown(
        "Welcome to the Interactive Stock Price Predictor! This project combines an in-depth analysis of Apple's stock values in the market with the power of a Long Short-Term Memory (LSTM) Recurrent Neural Network model.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "You can find the full work in [this Notebook in Google Colab](https://colab.research.google.com/drive/10nJ0_TKRVvA8gQ4eVLc0L7pxOHiCYW9c#scrollTo=azH5bR84ivPU)",
        unsafe_allow_html=True,
    )
    st.markdown(
        "You can find the full project in [this GitHub repository](https://github.com/rodrigo-dls/StockMarket-LSTM-Model/tree/master)",
        unsafe_allow_html=True,
    )

# ***************
# Dataset
# ***************

filename = "AAPL_2006-01-01_to_2018-01-01.csv"
data = get_data_with_date_index(filename)

with dataset:
    # Text
    st.header("Apple Stock Market Dataset")
    st.markdown(
        "I found this dataset on Kaggle. [Link to dataset](https://www.kaggle.com/dataset)"
    )

    # Table
    st.dataframe(data, height=230)

    # Visualization
    st.subheader("Dataset Visualization")
    fig_1 = st.container()
    with fig_1:
        fig = px.line(
            data.tail(100),
            x=data.tail(100).index,
            y=["High", "Low"],
            title="Evolution of High and Low stock price over the last 100 days of the dataset",
            line_shape="linear",
            line_dash_sequence=["solid"],
        )
        fig.update_layout(  # adjust reference box position
            legend=dict(x=0.75, y=0.15, title=None), xaxis_title="Date", yaxis_title="Value"
        )
        st.write(fig)

# ***************
# Features
# ***************

# Text description fo the data features
with features:
    st.header("The features")
    st.markdown("You can test the model using any of the following features:")
    st.markdown(
        """
            * **Open:** Represents the opening stock price on that date.
            * **High:** Represents the highest stock price on that date.
            * **Low:** Represents the lowest stock price on that date.
            * **Close:** Represents the closing stock price on that date.
        """
    )

# ***************
# Model training
# ***************

with model_training:
    # Text
    st.header("Build the model!")
    st.markdown("Here you get to choose the hyperparameters of the model and see how the performance changes!")

    # Collect feature and hyperparameters for the model training
    input_feature = st.selectbox(
        "Which feature should be used as the input feature?",
        options=["High", "Low", "Open", "Close"],
        index=0,
    )
    units_number = st.slider(
        "What should be the number of neurons of the model?",
        min_value=10,
        max_value=50,
        value=30,
        step=10,
    )
    n_epochs = st.selectbox(
        "How many times should the model iterate? (epochs)",
        options=[2, 5, 10, 20],
        index=0,
    )
    n_batch = st.selectbox(
        "How big should be the data batches to train the model?",
        options=[8, 16, 24, 32],
        index=3,
    )

# Split Data
training_set = data.loc[:"2016", input_feature]
validation_set = data.loc["2017":, input_feature]

# Normalize Data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_sc = sc.fit_transform(pd.DataFrame(training_set))

# Define data size for the model (Could be a function)
time_step = 60  # block size
m = len(training_set_sc)  # iteration limit

# Create X_train & Y_train
X_train = []
Y_train = []
for i in range(time_step, m):
    X_train.append(training_set_sc[i - time_step : i, 0])
    Y_train.append(
        training_set_sc[i, 0]
    )  # the data immediately following that of the block
X_train, Y_train = np.array(X_train), np.array(
    Y_train
)  # previous data format is returned to continue working
X_train = np.reshape(
    X_train, (X_train.shape[0], X_train.shape[1], 1)
)  # reshapes it to feed it to the model

# Define the model hyperparameters
input_dim = (X_train.shape[1], 1)  # (timesteps, features)
output_dim = 1
na = units_number  # number of units (neurons)

# Create the model
model = create_model(input_dim, output_dim, na)

# Train the model
model.fit(X_train, Y_train, epochs=n_epochs, batch_size=n_batch) # type: ignore

# ***************
# Prediction
# ***************

# Validation Data Normalization
validation_set_sc = sc.transform(pd.DataFrame(validation_set))

# Define data size for validation data
X_test = []
for i in range(time_step, len(validation_set_sc)):
    X_test.append(validation_set_sc[i - time_step : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict
prediction = model.predict(X_test)
prediction = sc.inverse_transform(prediction)  # inverts normalization

# ***************
# Metrics
# ***************

# Collect formatted validation and prediction datasets (Pandas Series)
Y_test = validation_set[time_step:] # get rigth size validation dataset
prediction_ds = pd.Series(prediction[:, 0], index=Y_test.index)  # gets rigth format prediction dataset

# Create metrics
metrics_dict = {
    "mse": {
        "name": "Mean Squared Error (MSE)",
        "description": "- Measures the average of the squared differences between predicted and actual values.\n - Lower values indicate better accuracy.",
        "value": mean_squared_error(Y_test, prediction_ds),
    },
    "mae": {
        "name": "Mean Absolute Error (MAE)",
        "description": "- Calculates the average of the absolute differences between predicted and actual values.\n - Provides a straightforward measure of error.",
        "value": mean_absolute_error(Y_test, prediction_ds),
    },
    "r2": {
        "name": "Coefficient of Determination (R^2)",
        "description": "- Represents the proportion of variance in the dependent variable predictable from the independent variable.\n - Higher values signify better predictive ability.",
        "value": r2_score(Y_test, prediction_ds),
    },
}

with metrics:
    # Text
    st.header("Metrics of the Model")
    display = st.empty() # creating a single-element container
    with display.container():
        m1, m2, m3 = st.columns(3) # create three columns
        for mcol, items in list(zip((m1,m2,m3),metrics_dict.items())): # fill in those three columns with respective metrics
            mcol.metric(
                label=items[1]['name'],
                value=round(number=items[1]["value"],ndigits=6),
                help=items[1]["description"],
                delta=float()
            )

# ***************
# Visualization of Results
# ***************

 # Create dataframe for viz
res = pd.DataFrame({"Validation": Y_test, "Prediction": prediction_ds})
with viz:
    # Comparison Validation vs Prediction
    fig = px.line(
        res,
        x=res.index,
        y=["Validation", "Prediction"],
        title="Comparison Validation vs Prediction",
        line_shape="linear",
        line_dash_sequence=["solid"],
    )
    fig.update_layout(  # adjust reference box position
        legend=dict(x=0.75, y=0.05, title=None), xaxis_title="Date", yaxis_title="Value"
    )
    st.plotly_chart(fig)

    # Percentage Difference between Validation and Predictions
    res["Error"] = (
        (res["Validation"] - res["Prediction"]) / res["Validation"]
    ) * 100  # calculate the percentage difference
    fig = px.line(
        res,
        x=res.index,
        y="Error",
        title="Percentage Difference between Validation and Predictions",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")  # add a horizontal line at y=0
    fig.update_layout(  # update layout
        xaxis_title="Date",
        yaxis_title="Percentage",
        title="Percentage Difference between Validation and Predictions",
        showlegend=False,
    )
    st.plotly_chart(fig)

# ***************
# Training Record
# ***************

# Create record_df cols name list
input_columns =  ["feature","n_epochs","n_batch","n_neurons"] # list of hyperparameter names
metric_columns = [] # list of metric names
for metric, meta in metrics_dict.items():
    metric_columns.append(metric) # get string metric names
record_columns =  ["timestamp"] + input_columns + metric_columns # create df columns name list

# Create record df
record_df = pd.DataFrame(columns = record_columns)

# <------- Add step here ------>
# Retrieve data from DB and add to record_df (use try/assert to avoid error for DataFrame not found)

record_df = add_new_training_data(input_feature, n_epochs, n_batch, units_number, metrics_dict, record_df)

with training_record:
    st.subheader("Training Record")
    st.markdown("Here you can view a record of each test. Feel free to experiment as many times as you'd like and then make an informed decision about the best parameters for your specific use-case.")

    # Show updated dataFrame
    st.dataframe(record_df)

# ***********************************
# Update DB with new training data
# ***********************************

# Get the current datetime in string format
timestamp = datetime.datetime.now()
formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S') # format the date and time as a string

# Get metrics values
metric_values = [] # list of metric values
for metric, meta in metrics_dict.items():
    metric_values.append(meta['value']) # get string metric names

# Connect Firestore DB
db = firestore.Client.from_service_account_json("firestore-key.json")

# Select collection to store data
doc_ref = db.collection('records').document()

# Load new data
doc_ref.set({
    'feature': input_feature,
    'mse': metric_values[0],
    'mae': metric_values[1],
    'r2': metric_values[2],
    'n_batch': n_batch,
    'n_epochs': n_epochs,
    'n_neurons': units_number,
    'timestamp': formatted_timestamp
})

# ***********************************
# Retrieve all data from DB (still on test stage)
# ***********************************

# Get all documents within the collection
docs = db.collection("records").stream()

with top_test:
    st.subheader("Testing here:")
    top_test.dataframe(record_df)

    # Iterate through the documents and display the fields
    i = 0 # iterator counter for testing purposes
    for doc in docs:
        st.markdown(f"Id del documento {doc.id}:")
        data = doc.to_dict()
        if data is not None:
            for key, value in data.items():
                if key in ("timestamp", "mae"):
                    st.markdown(f"{key}: {value}")
        else:
            st.warning("The document does not contain data or could not be read correctly.")
        i += 1 
        if i >= 5: break

    # # Then query to list all users
    # records_ref = db.collection('records')

    # for doc in records_ref.stream():
    #     st.write(doc.id)
    #     st.write('{} => {}'.format(doc.id, doc.to_dict()['mse']))