#import required modules:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

#loading the dataset:
data = pd.read_csv('csv_files/Mall_Customers.csv')

data.shape #result : (200, 5)

data.head()
# output will be:
#      CustomerID	Gender	Age	Annual Income (k$)	Spending Score (1-100)
#          0	       1	    Male	 19	  15	  39
#          1	       2	    Male	 21	  15	  81
#          2	       3	    Female	 23	  16	  77
#          4           5	    Female	 31	  17	  40


#so here are 5 columns
# we will take only relevent columns for our model which is required 
#so we will take only:
# 1.income
# 2.spending score

#as we also wwe will seperate our features:

X = data.iloc[:,[3,4]].values #this will sepeate our features from data

# we have taken column 3 and 4
# which is 3. Income, 4. Spending Score

#we will check if there is any missing value is present or not in our dataset:

data.isnull().sum()
#so there are no missing values
# so we will move forward

# now we will choose no of clusters so it can divide based on data:
# we will use a method to find the perfect no of clusters
#which is called 'Elbow-Method'

wcss = [] #wcss - within cluster sum of square
for i in range(1, 11):
    kmean = KMeans(n_clusters=  i, init= 'k-means++', random_state=2)
    kmean.fit(X)
    wcss.append(kmean.inertia_)
    
#now we will plot a graph for this 
# to view:

plt.plot(range(1,11), wcss) 
plt.title('The Elbow Method') 
plt.xlabel('No of Clusters')
plt.ylabel('Wcss')  
plt.show()

# so answer is : No of cluster is 5

#now we will train our model using kmeans:

kmean = KMeans(n_clusters=5, init='k-means++', random_state=2)
y = kmean.fit_predict(X)

accuracy = adjusted_rand_score()
#we will draw graph to visualize placemnet of clusters:
plt.scatter(X[y==0,0], X[y==0,1], s=50, c= 'red', label= 'cluster1')
plt.scatter(X[y==1,0], X[y==1,1], s=50, c= 'green', label='cluster2')
plt.scatter(X[y==2,0], X[y==2,1], s=50, c= 'black', label= 'cluster3')
plt.scatter(X[y==3,0], X[y==3,1], s=50, c= 'blue', label= 'cluster4') 
plt.scatter(X[y==4,0], X[y==4,1], s=50, c= 'cyan', label= 'cluster5')

plt.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:, 1], s=100, label= 'centeroid')
plt.title('cluster groups')
plt.xlabel('annual income')
plt.ylabel('spendig score')
plt.show()
#import required modules:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

#loading the dataset:
data = pd.read_csv('csv_files/Mall_Customers.csv')

data.shape #result : (200, 5)

data.head()
# output will be:
#      CustomerID	Gender	Age	Annual Income (k$)	Spending Score (1-100)
#          0	       1	    Male	 19	  15	  39
#          1	       2	    Male	 21	  15	  81
#          2	       3	    Female	 23	  16	  77
#          4           5	    Female	 31	  17	  40


#so here are 5 columns
# we will take only relevent columns for our model which is required 
#so we will take only:
# 1.income
# 2.spending score

#as we also wwe will seperate our features:

X = data.iloc[:,[3,4]].values #this will sepeate our features from data

# we have taken column 3 and 4
# which is 3. Income, 4. Spending Score

#we will check if there is any missing value is present or not in our dataset:

data.isnull().sum()
#so there are no missing values
# so we will move forward

# now we will choose no of clusters so it can divide based on data:
# we will use a method to find the perfect no of clusters
#which is called 'Elbow-Method'

wcss = [] #wcss - within cluster sum of square
for i in range(1, 11):
    kmean = KMeans(n_clusters=  i, init= 'k-means++', random_state=2)
    kmean.fit(X)
    wcss.append(kmean.inertia_)
    
#now we will plot a graph for this 
# to view:

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('Wcss')
plt.show()

# so answer is : No of cluster is 5

#now we will train our model using kmeans:

kmean = KMeans(n_clusters=5, init='k-means++', random_state=2)
y = kmean.fit_predict(X)

accuracy = adjusted_rand_score()
#we will draw graph to visualize placemnet of clusters:
plt.scatter(X[y==0,0], X[y==0,1], s=50, c= 'red', label= 'cluster1')
plt.scatter(X[y==1,0], X[y==1,1], s=50, c= 'green', label='cluster2')
plt.scatter(X[y==2,0], X[y==2,1], s=50, c= 'black', label= 'cluster3')
plt.scatter(X[y==3,0], X[y==3,1], s=50, c= 'blue', label= 'cluster4')
plt.scatter(X[y==4,0], X[y==4,1], s=50, c= 'cyan', label= 'cluster5')

plt.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:, 1], s=100, label= 'centeroid')
plt.title('cluster groups')
plt.xlabel('annual income')
plt.ylabel('spendig score')
plt.show()