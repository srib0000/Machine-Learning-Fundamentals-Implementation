import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd 
dataset = pd.read_csv("crime.csv")

# Load the dataset
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)
n, p = np.shape(data)

# Split the data into training and testing sets
size = int(0.75 * n)
highweights = [1, 10, 50, 0.1]
highrisknumbers = list()
lowrisknumbers = list()
sample_train = data[0:size, 0:-1]
label_train = data[0:size, -1]
sample_test = data[size:, 0:-1]
label_test = data[size:, -1]

# Create high and low-risk communities
highriskgroup1 = []
lowriskgroup1 = []

for i in range(size, n):
    if data[i, -1] > 0.8:
        highriskgroup1.append(data[i])
    else:
        lowriskgroup1.append(data[i])

highriskgroup1 = np.array(highriskgroup1)
lowriskgroup1 = np.array(lowriskgroup1)

# Create a linear regression model and fit it
model = linear_model.LinearRegression()
model.fit(sample_train,label_train)

# Function to compute high-risk community error
def highriskgroup():
    highriskrows, highriskcolumns = highriskgroup1.shape
    highriskdata = highriskgroup1[0:highriskrows, 0:-1]
    highrisklabel = highriskgroup1[0:highriskrows, -1]
    highpredvalues = model.predict(highriskdata)
    
# highweights = [1, 10, 50, 0.1]
    for i in highweights:
        meansqu = (highpredvalues - highrisklabel) ** 2
        weight = i * np.eye(len(meansqu))
        finalvalue = np.sum(weight * meansqu)
        highrisknumbers.append(finalvalue)
        print("highriskerror weight is equals to", i, "is", finalvalue)

# Function to compute low-risk community error
def lowriskgroup():
    lowriskrows, lowcolumns = lowriskgroup1.shape
    lowriskvalues = lowriskgroup1[0:lowriskrows, 0:-1]
    lowrisklabel = lowriskgroup1[0:lowriskrows, -1]
    lowpredvalues = model.predict(lowriskvalues)
    
    meansqu1 = (lowrisklabel - lowpredvalues) ** 2
    weight1 = np.eye(len(meansqu1)) * 1
    finalvalue1 = np.sum(weight1 * meansqu1)
    lowrisknumbers.append(finalvalue1)
    print("lowriskerror weight 1 is equals to", finalvalue1)

# Calculate errors for high and low-risk communities
def testingerrorvalues():
    print("\n\n")
    for i in range(len(highweights)):
      print("testing error wh= ",highweights[i], "and wl=1 is ",highrisknumbers[i]+lowrisknumbers[0])

# Calculate errors for high and low-risk communities and testing error values
highriskgroup()
lowriskgroup()
testingerrorvalues()




