import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, linear_model, svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import cluster

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors

#convert csv of data set to pandas dataframe`
df = pd.read_csv("winequality-white.csv", sep=';')

df.head()

df.describe()

corr = df.corr()

#plot correlation heatmap between all columns in dataframe
plt.subplots(figsize=(15,10))
ax = plt.axes()
ax.set_title("Wine Characteristic Correlation Heatmap")
sns.heatmap(corr, 
            xticklabels=corr.columns, 
            yticklabels=corr.columns, 
            annot=True, 
            cmap=sns.diverging_palette(220, 20, as_cmap=True))


#histogram of quality 
fig = plt.figure(figsize = (8,8))
plt.hist(df.quality,bins=6,alpha=0.5,histtype='bar',ec='black')
plt.title('Distribution of the Quality')
plt.xlabel('Quality')
plt.ylabel('Count')

#boxplot of quality and alcohol
plt.figure(figsize=(8,5))
sns.boxplot(x='quality',y='alcohol',data=df,palette='GnBu_d')
plt.title("Boxplot of Quality and Alcohol")
plt.show()

cols = ["total sulfur dioxide","free sulfur dioxide","residual sugar","fixed acidity","volatile acidity","alcohol","sulphates","pH","density", "citric acid", "chlorides"]

#separate data into training and test sets and choose state so that results are repeatable
X_train, X_test, y_train, y_test = train_test_split(df[cols], df["quality"], test_size=0.2, random_state=4) 

#declare and fit svm with rbf kernel to training set
clf = svm.SVC(kernel="rbf", gamma=1, C=1,
             decision_function_shape="ovo")
clf.fit(X_train,y_train)

ypred = clf.predict(X_test)

#get scores for how well the svm performs on test set 
print("Accuracy rbf kernel: %.2f"
      % clf.score(X_test, y_test))
print("nmi rbf kernel: %.2f"
      % normalized_mutual_info_score(y_test, ypred))
print(classification_report(y_test, ypred))


X = StandardScaler().fit_transform(df[cols])

#finding ideal eps for DBSCAN by calculating the distance to the nearest n points for each point, 
#sorting and plotting the results. Then we look to see where the change is most pronounce
#and select that as epsilon.
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title("finding optimal eps value for DBSCAN using elbow method")

params = {'n_clusters': 2, 'eps': 2, 'min_samples': 5}

spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], 
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")
dbscan = cluster.DBSCAN(eps=params['eps'], 
                        min_samples=params['min_samples'])

clustering_algorithms = (
        ('SpectralClustering', spectral),
        ('DBSCAN', dbscan),
    )
y_pred = {}

for name, algorithm in clustering_algorithms:
    y_pred[name] = algorithm.fit_predict(X)


#calculate number of true labels between 1-9 in each cluster and then choose the label which appears
#most frequently as the label of that cluster
avg_quality_spec = [[0, 0, 0, 0, 0, 0, 0, 0 ,0], [0, 0, 0, 0, 0, 0, 0]]
avg_quality_dbscan = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

for i, pred in enumerate(y_pred['DBSCAN']): 
    if pred == 0:
        avg_quality_dbscan[0][df['quality'].iloc[i] - 3] += 1
    elif pred == -1:
        avg_quality_dbscan[1][df['quality'].iloc[i] - 3] += 1
        
for i, pred in enumerate(y_pred['SpectralClustering']): 
    if pred == 0:
        avg_quality_spec[0][df['quality'].iloc[i] - 3] += 1
    elif pred == 1:
        avg_quality_spec[1][df['quality'].iloc[i] - 3] += 1
        

# in all the clusters, the quality of wine that appears the most in 6 so all elements
#in dataframe are labeled as 6
y_pred_ = [6 for i in range(4898)]
print("f1_score: %0.3f"
      % f1_score(df['quality'], y_pred_, average='weighted'))
print("accuracy_score: %.03f"
      % accuracy_score(df['quality'], y_pred_))
print("Mutual Information: %0.3f"
      % metrics.normalized_mutual_info_score(df['quality'], y_pred_))

#for labels in y_pred.values():
#    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#    print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#    print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels))

#dimension reduction with pca
pca = PCA(n_components=3).fit(X)
new_cor = pca.transform(X)

#plot projection of 3d pca onto two components 
fig = plt.figure(figsize=(15,10))

fig.add_subplot(1,3,1)
plt.scatter(new_cor[:,0], new_cor[:,1], marker='x', cmap='jet')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

fig.add_subplot(1,3,2)
plt.scatter(new_cor[:,0], new_cor[:,2], marker='x', cmap='jet')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 3")

fig.add_subplot(1,3,3)
plt.scatter(new_cor[:,1], new_cor[:,2], marker='x', cmap='jet')
plt.xlabel("PCA Component 2")
plt.ylabel("PCA Component 3")

#pearson correlation coefficient for all the pca components with quality
corr1 = scipy.stats.pearsonr(new_cor[:,0], df['quality'])
corr2 = scipy.stats.pearsonr(new_cor[:,1], df['quality'])
corr3 = scipy.stats.pearsonr(new_cor[:,2], df['quality'])
print(corr1)
print(corr2)
print(corr3)

#plot information content of each pca component
fig = plt.figure(figsize=(12,6))

plt.bar(np.arange(pca.n_components_), 100*pca.explained_variance_ratio_)
plt.title('Relative information content of PCA components')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance % ")

#dimension reduction with Isomap
iso = manifold.Isomap(n_neighbors=6, n_components=3).fit(X)
manifold_3Da = iso.transform(X)

#plot projection of 3d isomap onto two components
fig = plt.figure(figsize=(15,10))

fig.add_subplot(1,3,1)
plt.scatter(manifold_3Da[:,0], manifold_3Da[:,1], marker='x', cmap='jet')
plt.xlabel("Isomap Component 1")
plt.ylabel("Isomap Component 2")

fig.add_subplot(1,3,2)
plt.scatter(manifold_3Da[:,0], manifold_3Da[:,2], marker='x', cmap='jet')
plt.xlabel("Isomap Component 1")
plt.ylabel("PCA Component 3")

fig.add_subplot(1,3,3)
plt.scatter(manifold_3Da[:,1], manifold_3Da[:,2], marker='x', cmap='jet')
plt.xlabel("Isomap Component 2")
plt.ylabel("Isomap Component 3")

plt.show()

#pearson correlation coefficient for all the Isomap components with quality
corr1 = scipy.stats.pearsonr(manifold_3Da[:,0], df['quality'])
corr2 = scipy.stats.pearsonr(manifold_3Da[:,1], df['quality'])
corr3 = scipy.stats.pearsonr(manifold_3Da[:,2], df['quality'])
print(corr1)
print(corr2)
print(corr3)




