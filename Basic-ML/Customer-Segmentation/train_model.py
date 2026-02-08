import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
import joblib

# Loading DataSet
df = pd.read_csv("Mall_Customers.csv")

df.info

df.isnull().sum()

df.describe()

df.shape

df.duplicated().sum()

df.head()

# Selecting important features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method:
# Trying different numbers of clusters(1-20):

scores = []

for i in range(1, 21):
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(X)
    scores.append(model.inertia_)


# Plot:
plt.plot(range(1, 21), scores, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")

plt.show()

# from the above elbow plot, we can conclude n_clusters=5 is the best value

# Train final model:
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Save model : joblib faster,consumes less RAM,meant for scikit-learn 
joblib.dump(kmeans, "kmeans_model.pkl")

import sklearn
print(sklearn.__version__)


