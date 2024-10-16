import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error



#create objects
#minmax scaler
scaler=MinMaxScaler()
#define model
lm=LinearRegression()

#drop data function
def drop_data(df,y_column):
    #first we have to drop data
    y_df=df[[y_column]]
    df=df.drop('Email',axis=1)
    df=df.drop('Address',axis=1)
    df=df.drop('Avatar',axis=1)
    df=df.drop(y_column,axis=1)
    return y_df,df

#creates other columns
def other_columns(df):
    df['Session+App_time'] = df['Avg. Session Length'] + df['Time on App']
    df['Session+Web_time'] = df['Avg. Session Length'] + df['Time on Website']
    df['Session+Length'] = df['Avg. Session Length'] + df['Length of Membership']
    df['App_time+Web_time'] = df['Time on App'] + df['Time on Website']
    df['App_time+length'] = df['Time on App'] + df['Length of Membership']
    df['WebTime+Length'] = df['Time on Website'] + df['Length of Membership']
    df['all'] = df['Avg. Session Length'] + df['Time on App'] + df['Time on Website'] + df['Length of Membership']
    df['Session+App+Web'] = df['Avg. Session Length'] + df['Time on App'] + df['Time on Website']
    df['App+Web+Membership'] = df['Time on App'] + df['Time on Website'] + df['Length of Membership']
    df['Web+Length+Session'] = df[['Time on Website', 'Length of Membership', 'Avg. Session Length']].sum(axis=1)  # Corrected line
    df['Length+Session+App'] = df['Length of Membership'] + df['Avg. Session Length'] + df['Time on App']
    return df

#split data with sklearn
def sklearn_split_data(data,test_ratio):
    #now insert into split data from sklearn
    X_train,X_test,y_train,y_test=train_test_split(data,y_df,test_size=test_ratio,random_state=42)
    return X_train,X_test,y_train,y_test

#read csv
df=pd.read_csv('Ecommerce Customers')

#now we have to drop columns
y_df,df=drop_data(df,'Yearly Amount Spent')

#normalize data
normalized_df=scaler.fit_transform(df)
normalized_df=pd.DataFrame(columns=['Avg. Session Length','Time on App','Time on Website','Length of Membership'],data=normalized_df)

#okay now make other columns
df=other_columns(df)



#split data
X_train,X_test,y_train,y_test=sklearn_split_data(df,0.2)

#fit data
lm.fit(X_train,y_train)

#print coeficients
#print(lm.coef_)

#now make predcitions
predictions=lm.predict(X_test)

#now assess predictions
print(mean_absolute_error(y_test,predictions))
print(mean_squared_error(y_test,predictions))

#plot residuals
residuals=y_test-predictions
