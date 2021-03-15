import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime, calendar, time

style.use('ggplot')
quandl.ApiConfig.api_key = "z4zWwnXEQRMwzSz9RvQw"
#df = quandl.get("WIKI/GOOGL")
#Weather International Stock
#https://www.quandl.com/data/SSE/0WE-WEATHERFORD-INTERNATIONAL-STOCK-WKN-A116P6-ISIN-IE00BLNN3691-0WE-Stuttgart
df1 = quandl.get("SSE/0WE")

#df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
#df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
#df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df1['Elevation_PC'] = (df1['High'] - df1['Low']) / df1['High'] * 100.0
print df1.head()

df1 = df1[['High', 'Low', 'Elevation_PC']]

forecast_col = 'Elevation_PC'

df1.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df1)))
df1['label'] = df1[forecast_col].shift(-forecast_out)

X = np.array(df1.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df1.dropna(inplace=True)
y = np.array(df1['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df1['Forecast'] = np.nan

last_date = df1.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df1.loc[next_date] = [np.nan for _ in range(len(df1.columns)-1)]+[i]

df1['Elevation_PC'].plot()
df1['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig("out_"+str(calendar.timegm(time.strptime('Jul 9, 2009 @ 20:02:58 UTC', '%b %d, %Y @ %H:%M:%S UTC')))+".pdf")
