import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

#define model
lm=LinearRegression()

#EDA
#seaborn plots
def seasborn_plots(plot,x_name,y_name,df_data):
    if plot=="jointplot":
        sns.jointplot(x=x_name,y=y_name,data=df_data,alpha=0.5)
    elif plot=="pairplot":
        sns.pairplot(df_data,kind="scatter",plot_kws={'alpha':0.5})
    elif plot=="lmplot":
        sns.lmplot(x=x_name,y=y_name,data=df_data,scatter_kws={'alpha':0.5})
    elif plot=="displot":
        sns.displot(df_data,bins=30)
    #show data
    plt.show()


#function for splitting
def sklearn_split_data(data,y_column,test_ratio):
    #make dataframe for y
    df=data[[y_column]]
    #now drop column from data
    data=data.drop(y_column,axis=1)
    data=data.drop('Email',axis=1) 
    data=data.drop('Address',axis=1)
    data=data.drop('Avatar',axis=1) 
    #now insert into split data from sklearn
    X_train,X_test,y_train,y_test=train_test_split(data,df,test_size=test_ratio,random_state=42)
    return X_train,X_test,y_train,y_test

#read data
df=pd.read_csv('Ecommerce Customers')

#plot data
#seasborn_plots("pairplot",'Length of Membership','Yearly Amount Spent',df)

#split data
X_train,X_test,y_train,y_test=sklearn_split_data(df,'Yearly Amount Spent',0.2)

#fit data
lm.fit(X_train,y_train)

#print coeficients
print(lm.coef_)

#now make predcitions
predictions=lm.predict(X_test)

#now assess predictions
print(mean_absolute_error(y_test,predictions))
print(mean_squared_error(y_test,predictions))

#plot residuals
residuals=y_test-predictions
seasborn_plots('displot','a','b',residuals)





