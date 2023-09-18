import streamlit as st
# Data manipulation
import numpy as np
import pandas as pd
# Visualization
import plotly.express as px
# Data normalization
from sklearn.preprocessing import MinMaxScaler 
# Kearning models
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Page configuration
st.set_page_config(
    page_title="Stock Market Predictor App",
    page_icon="📈",
    layout="wide",
    # initial_sidebar_state="expanded"
)

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# Header
with header:
    st.title("Interactive Stock Price Predictor: Customize Your Forecasts")
    st.markdown("Welcome to the Interactive Stock Price Predictor! This project combines an in-depth analysis of Apple's stock values in the market with the power of a Long Short-Term Memory (LSTM) Recurrent Neural Network model.", unsafe_allow_html=True)
    st.markdown("You can find the full work in [this Notebook in Google Colab](https://colab.research.google.com/drive/10nJ0_TKRVvA8gQ4eVLc0L7pxOHiCYW9c#scrollTo=azH5bR84ivPU)", unsafe_allow_html=True)
    st.markdown("You can find the full project in [this GitHub repository](https://github.com/rodrigo-dls/StockMarket-LSTM-Model/tree/master)", unsafe_allow_html=True)

# Dataset
data = pd.read_csv('AAPL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'] )
with dataset:
    st.header("Apple Stock Market Dataset")
    st.markdown("I found this dataset on Kaggle. [Link to dataset](https://www.kaggle.com/dataset)")
    
    # Table
    st.dataframe(data, height= 230)
    
    # Visualization
    st.subheader("Dataset Visualization")
    fig = px.line(
                data.tail(100), x=data.tail(100).index, y=['High', 'Low'], 
                title='Evolution of High and Low stock price over the last 100 days of the dataset',
                line_shape='linear', line_dash_sequence=['solid'])
    fig.update_xaxes(title_text='Date')
    st.plotly_chart(fig)

# Features
with features:
    st.header("The features")
    st.markdown("You can test the model using any of the following features:")
    st.markdown(
        '''
            * **Open:** Represents the opening stock price on that date.
            * **High:** Represents the highest stock price on that date.
            * **Low:** Represents the lowest stock price on that date.
            * **Close:** Represents the closing stock price on that date.
        '''
    )


with model_training:
    st.header("Time to train the model!")
    st.markdown(
        "Here you get to choose the hyperparameters of the model and see how the performance changes!"
    )
    sel_col, disp_col = st.columns(2)
    units_number = sel_col.slider(
        "What should be the number of neurons of the model?",
        min_value=10,
        max_value=50,
        value=30,
        step=10,
    )
    # n_estimators = sel_col.selectbox(
    #     "How many trees should there be?", options=[100, 200, 300, "No limit"], index=0
    # )
    input_feature = sel_col.selectbox(
        "Which feature should be used as the input feature?", options=['High', 'Low','Open','Close'], index=0
    )
    
# Data Split
training_set = data.loc[:'2016','High']
validation_set = data.loc['2017':,'High']

# Data Normalization
sc = MinMaxScaler(feature_range=(0,1))
training_set_sc = sc.fit_transform(pd.DataFrame(training_set))
validation_set_sc = sc.transform(pd.DataFrame(validation_set))

# Define data size for the model
time_step = 60  # block size
m = len(training_set_sc) # iteration limit
X_train = []
Y_train = []
for i in range(time_step, m):
  X_train.append(training_set_sc[i-time_step:i,0])
  Y_train.append(training_set_sc[i,0])  # the data immediately following that of the block
X_train, Y_train = np.array(X_train), np.array(Y_train) # previous data format is returned to continue working
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # reshapes it to feed it to the model

# Creates the model
input_dim = (X_train.shape[1],1) # (timesteps, features)
output_dim = 1
na = units_number # number of units (neurons)

model = Sequential()
model.add(LSTM(units=na, input_shape=input_dim))    # LSTM Layer
model.add(Dense(units=output_dim))  # Output Layer (without activation function)
model.compile(optimizer='rmsprop', loss='mse')  # Compilation

