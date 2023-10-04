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

# tf.random.set_seed(4)

# Page configuration
st.set_page_config(
    page_title="Stock Market Predictor App",
    page_icon="📈",
    layout="wide"
)

# Authenticate to Firestore with the JSON account key.
db = firestore.Client.from_service_account_json("firestore-key.json")

# Create a reference to the Google post.
doc_ref = db.collection("records").document("tjuxpBn3KLlBPyMU0qio")

# Then get the data at that reference.
doc = doc_ref.get()

# # Cargar el dataset existente si lo hay
# try:
#     df = pd.read_csv("training_records.csv")
# except FileNotFoundError:
#     # Si el archivo no existe, lo creamos con las columnas
#     df = pd.DataFrame(columns=["feature", "n_epochs", "n_batch", "n_neurons"])




@st.cache_data
def get_data_with_date_index(filename):
    data = pd.read_csv(filename, index_col="Date", parse_dates=["Date"])
    return data


@st.cache_data
def register_training_data(input_feature, n_epochs, n_batch, units_number, metrics_dict, record_df):
    # Get the current date and time
    timestamp = datetime.datetime.now()
    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S') # format the date and time as a string

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

    doc_ref = db.collection('records').document()
    doc_ref.set({
        'feature': input_feature,
        'mse': new_metrics[0],
        'mae': new_metrics[1],
        'r2': new_metrics[2],
        'n_batch': n_batch,
        'n_epochs': n_epochs,
        'n_neurons': units_number,
        'timestamps': formatted_timestamp
    })

    return record_df

# @st.cache_data
# def update_df(new_df, old_df):
#     if not new_df.loc[len(record_df.index)-1] == old_df.loc[len(record_df.index)-1]:

# @st.cache_data
def create_model(input_dim, output_dim, na):
    model = Sequential()
    model.add(LSTM(units=na, input_shape=input_dim))    # LSTM Layer
    model.add(Dense(units=output_dim))  # Output Layer (without activation function)
    model.compile(optimizer='rmsprop', loss='mse')  # Compilation
    return model

# ***************
# App Strutcture
# ***************
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
data = get_data_with_date_index("AAPL_2006-01-01_to_2018-01-01.csv")
with dataset:
    st.header("Apple Stock Market Dataset")
    st.markdown(
        "I found this dataset on Kaggle. [Link to dataset](https://www.kaggle.com/dataset)"
    )

    # Table
    st.dataframe(data, height=230)

    fig_col1, fig_col2 = st.columns(2)
    # Visualization
    st.subheader("Dataset Visualization")
    with fig_col1:
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

    # with fig_col2:
    #     fig = px.line(
    #         data.tail(100),
    #         x=data.tail(100).index,
    #         y=["Open", "Close"],
    #         title="Evolution of Open and Close stock price over the last 100 days of the dataset",
    #         line_shape="linear",
    #         line_dash_sequence=["solid"],
    #     )
    #     fig.update_layout(  # adjust reference box position
    #         legend=dict(x=0.75, y=0.15, title=None), xaxis_title="Date", yaxis_title="Value"
    #     )
    #     st.write(fig)

# ***************
# Features
# ***************
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
    st.header("Build the model!")
    st.markdown("Here you get to choose the hyperparameters of the model and see how the performance changes!")

    # Set feature and hyperparameters
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

# Define data size for the model
time_step = 60  # block size
m = len(training_set_sc)  # iteration limit
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

# Create the model
input_dim = (X_train.shape[1], 1)  # (timesteps, features)
output_dim = 1
na = units_number  # number of units (neurons)

model = create_model(input_dim, output_dim, na)

# Train the model
model.fit(X_train, Y_train, epochs=n_epochs, batch_size=n_batch)
# model = train_model(model, X_train, Y_train, n_epochs, n_batch)


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
Y_test = validation_set[time_step:] # gets rigth size validation dataset
prediction_ds = pd.Series(prediction[:, 0], index=Y_test.index)  # gets rigth format prediction dataset

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
    st.subheader("Metrics of the Model")
    # m1, m2, m3 = st.columns(3) # create three columns
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
# # Store selected parameters
# new_inputs = [
#         input_feature,
#         n_epochs,
#         n_batch,
#         units_number
#     ]

# Store new metrics
# new_metrics = []
metric_columns = []
for metric, meta in metrics_dict.items():
    # new_metrics.append(meta['value'])
    # st.markdown(f"{new_metrics}")
    metric_columns.append(metric)
    # st.markdown(f"{metric_columns}")
    # st.markdown("\n")


# Create df columns array
input_columns =  ["feature","n_epochs","n_batch","n_neurons"]

record_columns =  ["timestamp"] + input_columns + metric_columns

# Create record df
record_df = pd.DataFrame(columns = record_columns)
# Cargar dataset con los records: record_df = load_record_df()

# Create new record
# new_row = new_inputs + new_metrics
# record_df.loc[len(record_df.index)] = new_row   # adds new row to df

record_df = register_training_data(input_feature, n_epochs, n_batch, units_number, metrics_dict, record_df)
# new_row = pd.DataFrame(
#     {
#         "feature": [input_feature],
#         "n_epochs": [n_epochs],
#         "n_batch": [n_batch],
#         "n_neurons": [units_number],
#     }
# )
# 
# df = df.append(new_row, ignore_index=True)


with training_record:
    st.subheader("Training Record")
    st.markdown("Here you can view a record of each test. Feel free to experiment as many times as you'd like and then make an informed decision about the best parameters for your specific use-case.")
    # for metric, meta in metrics_dict.items():
    #     st.markdown(f"{meta['name']}: {meta['value']}")

    # Mostrar el DataFrame actualizado
    st.dataframe(record_df)

top_test.dataframe(record_df)

with top_test:
    # Then query to list all users
    records_ref = db.collection('records')

    for doc in records_ref.stream():
        st.write('{} => {}'.format(doc.id, doc.to_dict()['mse']))