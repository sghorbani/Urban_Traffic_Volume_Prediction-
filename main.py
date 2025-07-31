import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display

columns=['holiday','temp','rain_1h','snow_1h','clouds_all','weather_main','weather_description','date_time','traffic_volume']
df = pd.read_csv('./data/Metro_Interstate_Traffic_Volume.csv' )

df.head
print (f"rows {df.shape[0]} - columns {df.shape[1]} .")

print("missing values per column:")
print(df.isnull().sum())
print('-------------------------')


print("data frame info:")
df.info()

print("data frame describe:")
print(df.describe())
print('-------------')
 
print("-----df.dtypes----")
print(df.dtypes )

print("-----df.columns.tolist----")
print(df.columns.tolist())

print("-----df.describe().T----")
print(df.describe().T)


categorical_cols = ['holiday', 'weather_main', 'weather_description', 'date_time']
for col in categorical_cols:
    print(f"----column: {col}")
    print(df[col].value_counts().head(10))

numeric_cols=df.select_dtypes(include=['float64', 'int64']).columns   
for col in numeric_cols:
    print(f"------column: {col}")
    print(df[col].describe())
    print(df[col].isnull().sum())
    print(df[col].nunique())

df['date_time']=pd.to_datetime(df['date_time'])    

df['hour']=df['date_time'].dt.hour
df['weekday']=df['date_time'].dt.weekday
df['month']=df['date_time'].dt.month

def month_to_season(month):
    if month in [3,4,5]:
        return 'spring'
    elif month in [6,7,8]:
        return 'summer'
    elif month in [9,10,11]:
        return 'autumn'
    else:
        return 'winter'

df['season']=df['month'].apply(month_to_season) 

df['temp_c']=df['temp']- 273.15

df = df.drop(['date_time', 'temp'], axis=1)


print('---------------one-hot------------------')

categorical_cols = ['holiday', 'weather_main', 'weather_description', 'season']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(df.head())
print('---------------outlires------------------')
plt.figure(figsize=(8,4))
df['traffic_volume'].hist(bins=40)
plt.title('Traffic Volume Distribution')
# plt.show()

df = df[(df['traffic_volume'] > 50) & (df['traffic_volume'] < 7000)]
 

print('---------------prepare train data------------------')

X = df.drop(['traffic_volume'], axis=1)
y = df['traffic_volume']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))


plt.figure(figsize=(8,4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Real Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.title("Actual vs Predicted")
plt.show()

coefs = pd.Series(lr.coef_, index=X.columns)
print(coefs.sort_values(ascending=False))
