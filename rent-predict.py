# * Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# * LOADING DATA
# * Data is receive from https://www.kaggle.com/
data = pd.read_csv("House_Rent_Dataset.csv")
data.head()
data.info()

# * checking distribution of the target variables
sns.distplot(data['Rent'], color='red')
plt.axvline(x=data['Rent'].mean(), color='red', linestyle='--', linewidth=2)
plt.title('Rent of House')

# * Check missing values
print("Sum of the null values:")
print(data.isna().sum())

# Descriptive Analysis
describe = data.describe()

# * Correlation
corr = data.corr()
plt.figure(figsize=(30, 9))
sns.heatmap(corr, annot=True)

# Check columns data type
data.dtypes

# BHK
sns.boxplot(y="BHK", data=data)
plt.show()

# Count the categories of Area Type column
sns.countplot(x=data['Area Type'])

# Plot count of the City column's categories
sns.countplot(x=data['City'])

data['City'].unique()
# Count plot of between City and Rent
plt.subplots(figsize=(15, 7))
ax = sns.boxplot(x='City', y='Rent', data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.show()

# Count the categories of Furnishing Status column
sns.countplot(x=data['Furnishing Status'])
data['Furnishing Status'].value_counts()

# Count the categories of Tenant Preferred column
sns.countplot(x=data['Tenant Preferred'])
data['Tenant Preferred'].value_counts()

# Count the categories of Point of Contact column
sns.countplot(x=data['Point of Contact'])


# * ---> Pre-processing of the Data
# City column is grouped by 'Rent' col value mean according to its own area
cityRent = data.groupby(['City'])['Rent'].mean()
data['City'] = data['City'].apply(lambda value: cityRent[value])

# Drop Built Area which is value of 'Area Type' col
# count of Built Area  = 2, so it is inadequate
data.drop(data[data['Area Type'] == 'Built Area'].index, inplace=True)

# convert object to int


def floor(value):
    if 'Ground' in value:
        return 0
    elif 'Upper Basement' in value:
        return -1
    elif 'Lower Basement' in value:
        return -2
    return int(value.split(" ")[0])


data['Floor'] = data['Floor'].apply(lambda x: floor(x))
sns.countplot(x=data['Floor'])


# Contact Builder =1 , remove it
data.drop(data[data['Point of Contact'] ==
          'Contact Builder'].index, inplace=True)

# drop redundant column
to_drop = ["Posted On", "Area Locality"]
data = data.drop(to_drop, axis=1)


# * ---> Modelling
x = data.drop(['Rent'], axis=1)
y = data['Rent']

# Get all categorical fields which are object dtype and then put in a list
categorical_col = [col for col in x.columns if x[col].dtype == "object"]

# Label Encoding the object dtypes ( categorical field) and use LabelEncoder()
LE = LabelEncoder()
for i in categorical_col:
    x[i] = x[[i]].apply(LE.fit_transform)

# Scaling ( scale 0-1 ) for standardization
scaler = StandardScaler()
scaler.fit(x)
scaled_data = pd.DataFrame(scaler.transform(x), columns=x.columns)
scaled_data.head()
x = scaled_data

# Split the data test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# put the models in the dictionary
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor()
}
# put models in the array
model_results = []
model_names = []

# training the model
for name, model in models.items():
    a = model.fit(x_train, y_train)
    predicted = a.predict(x_test)
    score = np.sqrt(mean_squared_error(y_test, predicted))
    model_results.append(score)
    model_names.append(name)

    # creating dataframe
    df_results = pd.DataFrame([model_names, model_results])
    df_results = df_results.transpose()
    # RMSE: Root mean square error
    df_results = df_results.rename(
        columns={0: 'Model', 1: 'RMSE'}).sort_values(by='RMSE', ascending=False)


print(df_results)

# RandomForestRegressor
RFR = RandomForestRegressor()
model = RFR.fit(x_train, y_train)
predicted = model.predict(x_test)
score = np.sqrt(mean_squared_error(y_test, predicted))
print(score)
