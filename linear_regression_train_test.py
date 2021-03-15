import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = "z4zWwnXEQRMwzSz9RvQw"
df = quandl.get('WIKI/GOOGL')
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0;
df['PCT_CHANGE'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0;
df = df[['Adj. Open','HL_PCT','PCT_CHANGE','Adj. Close','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X_lately = X[-forecast_out:]
y = np.array(df['label'])

X = preprocessing.scale(X)
df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf1 = LinearRegression(n_jobs=10)
clf1.fit(X_train, y_train)
accuracy_Linear = clf1.score(X_test, y_test)
print "Linear>>>"+str(accuracy_Linear)

clf2 = svm.SVR(kernel='poly', shrinking= True)
clf2.fit(X_train, y_train)
accuracy_Vector = clf2.score(X_test, y_test)
print "Vector>>>"+str(accuracy_Vector)

clf3 = svm.SVR()
clf3.fit(X_train, y_train)
accuracy_Vector_poly = clf3.score(X_test, y_test)
print "Vector Poly>>>"+str(accuracy_Vector_poly)
