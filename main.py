########################README###########################
# Step 1) Click 'Open in Playground'                    #
# Step 2) Click play button and the 'run anyway' popup  #
# Step 3) Click arrow tab on right and click files      #
# Step 4) Click Refresh                                 #
# Step 5) Download Model.csv                            #
#########################################################

"""
    File name: q2lab.ipynb
    Author: Daniel Ko
    Date created: 1/12/2019
    Date last modified: 1/14/2019
    Python Version: 3.6.7
"""

from scipy.optimize import curve_fit
import pandas as pd
from sympy import *
from sklearn.metrics import mean_absolute_error


#raw csv data is converted into a pandas.DataFrame 
#https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.html
data = pd.read_csv("https://gist.githubusercontent.com/hdko/8932e5efb186503f68bc9c3f7b50bedc/raw/c5af13e5e942d0e8ba10f4e8d822fbcab6fa6750/gistfile1.txt")
data.columns = ['wavelength','red12','green12','blue12','blue9','blue6','blue3','yellow12','mixture3','protoModel']


"""
code for non linear regression
"""

def func(i, r, b, y):
  """Model function of mixture3 for non-linear regression

  Args:
    i: Index of a wavelength
    r, b, y: Coefficients for red12, blue12, and yellow12 respectively that we are trying to optimize   
  Returns:
    Absorption of our model
  """
  return r * data['red12'][i] + b * data['blue12'][i] +  y * data['yellow12'][i]




#non linear regression is run here
popt,pcov = curve_fit(func, list(range(0, 455)), data['mixture3'])
#popt: Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
#pcov: Estimated covariance of popt
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
#print(pcov) #values close to 0 means non linear (?)

#create a model of mixture3 using non linear regression data 
model_NLR =  data['red12'] * popt[0] + data['blue12']* popt[1] + data['yellow12']* popt[2]
print("nonlinear regression")
print("mean_absolute_error of non linear regression model",mean_absolute_error(data.mixture3,model_NLR))
print("?:",popt[0]," ß:",  popt[1], " ?:", popt[2], "\n")


"""
code for max/peak values method
"""


# x, y, z are coefficients for red12, blue12, and yellow12 respectively    
x, y, z = symbols('x y z')


#indexs for max values
maxRed = data.red12[data.red12 == max(data.red12)].index[0]
maxBlue = data.blue12[data.blue12 == max(data.blue12)].index[0]
maxYellow = data.yellow12[data.yellow12 == max(data.yellow12)].index[0]


t = list(linsolve([
                   Eq(x*data['red12'][maxRed] + y*data['blue12'][maxRed] + z*data['yellow12'][maxRed], data['mixture3'][maxRed]), 
                   Eq(x*data['red12'][maxBlue] + y*data['blue12'][maxBlue] + z*data['yellow12'][maxBlue], data['mixture3'][maxBlue]), 
                   Eq(x*data['red12'][maxYellow] + y*data['blue12'][maxYellow] + z*data['yellow12'][maxYellow], data['mixture3'][maxYellow])],                                                                   
                    [x, y, z]))

    
#create a model of mixture3 using max values
print("system of equations from maximum values")
model_peak =  data['red12'] * t[0][0] + data['blue12']* t[0][1] + data['yellow12']* t[0][2]
print("mean_absolute_error of max model",mean_absolute_error(data.mixture3,model_peak))
print("?:",t[0][0]," ß:",  t[0][1], " ?:", t[0][2])

print("")
print("mean_absolute_error of between models",mean_absolute_error(model_NLR,model_peak))
print("mean_absolute_error of blue6 model", mean_absolute_error(data.blue6,data.blue12/2))


"""
Convert into csv :)
"""

modelCSV = pd.DataFrame({"wavelength(nm)": data['wavelength'], "model_NLR": model_NLR, "model_peak": model_peak})
modelCSV.to_csv(r'Model.csv')
