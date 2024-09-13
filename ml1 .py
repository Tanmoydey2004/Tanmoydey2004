import pandas as pd
import matplotlib.pyplot as 
import seaborn as sns 
sns.set()
from skl.cluster import KMeans
from sklearn import preprocessing 

<!--[2]-->
data = pd.read_csv('Example.csv')

<!--[3]-->
plt.scatter(data['Satisfaction],data['Loyalty'])
plt.xlable('Satisfaction')
plt.ylable('Loyalty')

<!--[4]-->            
x = preprocessing.scale(data)
kmeans = KMeans(4)
kmeans.fit(x)

<!--[5]-->
plt.scatter(data['Satisfaction],data['Loyalty'],c=kmeans.fit_predidict(x),cmap='rainbow') 
plt.xlable('Satisfaction')
plt.ylable('Loyalty')

<!--[6]-->









<!---
Tanmoydey2004/Tanmoydey2004 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
