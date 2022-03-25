
from audioop import add
from copy import deepcopy
from sqlite3 import Date
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import streamlit as st
from datetime import date
import yfinance as yf
import plotly
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import pandas as pd
import sklearn
import datetime
import numpy as np
import tkinter as tk

#DATE
START = "2019-01-02"
TODAY = date.today().strftime("%Y-%m-%d")
 #TITLE
st.title('Stock Price Prediction System')
st.write("""
This app predicts the Stock Price**!
""")
st.write('---')
# 

# #DROPDOWN
stocks = ('BBD', 'RELI', 'TSLA', 'AAPL', 'PEP',
          'DIS', 'INFY', 'AA', 'SBIN.NS', 'HDFCBANK.NS',
          'M&M.NS', 'TATAMOTORS.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS',
          'BAJAJ-AUTO.NS', 'DABUR.NS', 'WRX-USD', 'MRF.NS','ASIANPAINT.BO',
          'ASIANPAINT.NS', 'ADANIPORTS.BO', 'AXISBANK.BO', 'KOTAKBANK.BO', 'KOTAKBANK.NS',
          'NESTLEIND.NS', 'VEDL.BO', 'ZOMATO.NS', 'BERGEPAINT.BO', 'BLUESTARCO.NS', 'NMDC.BO',
          'VOLTAS.BO', 'HCLTECH.BO', 'TATAPOWER.NS', 'USHAMART.NS', 'KALYANKJIL.BO',
          'KALYANKJIL.NS', 'BPCL.NS', 'IOC.NS', 'IRCTC.NS', 'MARUTI.BO' )
#selected_stocks = st.multiselect("Select Stocks", stocks)
selected_stocks = st.selectbox("Select Stocks", stocks)

st.write('---')
st.write("""Click here to know more about symbols""")
link = '[Symbols](https://stockanalysis.com/stocks/)'
st.markdown(link, unsafe_allow_html=True)

#

def nearest_business_day(DATE: datetime.date):
    
    #Takes a date and transform it to the nearest business day
    
    if DATE.weekday() == 5:
        DATE = DATE - datetime.timedelta(days=1)

    if DATE.weekday() == 6:
        DATE = DATE + datetime.timedelta(days=1)
    return DATE
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date
YESTERDAY=datetime.date.today()-datetime.timedelta(days=1)
YESTERDAY = nearest_business_day(YESTERDAY) #Round to business day

DEFAULT_START=YESTERDAY - datetime.timedelta(days=700)
DEFAULT_START = nearest_business_day(DEFAULT_START)

START = sub_columns[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY - datetime.timedelta(days=1))
END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)

START = nearest_business_day(START)
END = nearest_business_day(END)

# ---------------stock selection------------------
#STOCKS = np.array([ "GOOG", "GME", "FB","AAPL",'TSLA'])  
#SYMB = window_selection_c.selectbox("select stock", STOCKS)


# #SLIDER
n_years = st.slider("Number of Years", 1, 10)
period = n_years * 365
# 
# #RAW DATA
# # st.cache
# # The actual bytecode that makes up the body of the function
# # Code, variables, and files that the function depends on
# # The input parameters that you called the function with
# # If this is the first time Streamlit has seen these items, with these exact values, and in this exact combination, it runs the function and stores the result in a local cache.
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
# 
data_load_state = st.text("Loading Data...")
data = load_data(selected_stocks)
data_load_state.text("Loading Data...done!")
# 
st.subheader("Raw Data")
st.write(data)
# 
#fig = go.Figure()
#data['Close'].plot(figsize=(36,18),color='#002699',alpha=0.8,legend=True)
#fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
#fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
#plt.xlabel("Date",fontsize=12,fontweight='bold',color='black')
#plt.ylabel('Price',fontsize=12,fontweight='bold',color='black')
#plt.title("Stock price for S&P 500",fontsize=18)
#plt.show()
#plt.savefig('demo.png', bbox_inches='tight')
#fig.layout.update(title_text='Time Series data with Rangeslider')
#st.line_chart(fig)

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Raw Data graph with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()



# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#yhat = forecast
#A trend is the general direction at which the stock is moving
m = Prophet(daily_seasonality = True)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Predicted data')
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

#from sklearn.metrics import mean_absolute_error
#loss = mean_absolute_error(y_true=forecast["yhat"])
#loss