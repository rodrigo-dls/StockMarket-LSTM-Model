import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
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
with dataset:
    st.header("Apple Stock Market Dataset")
    st.markdown("I found this dataset on Kaggle. [Link to dataset](https://www.kaggle.com/dataset)")
    
    # Table
    data = pd.read_csv('AAPL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'] )
    st.dataframe(data, height= 230)
    
    # Visualization
    st.subheader("Dataset Visualization")
    fig = px.line(data.tail(100), x=data.tail(100).index, y='High', title='Highest stock price of the last 100 days of the dataset', line_shape='linear', line_dash_sequence=['solid'])
    fig.update_xaxes(title_text='Date')
    fig.update_traces(line=dict(color='orange'))
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
    max_depth = sel_col.slider(
        "What should be the max_depth of the model?",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
    )
    n_estimators = sel_col.selectbox(
        "How many trees should there be?", options=[100, 200, 300, "No limit"], index=0
    )
    input_feature = sel_col.text_input(
        "Which feature should be used as the input feature?", "High"
    )
