# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope, MinCovDet, EmpiricalCovariance
from pyod.models.iforest import IsolationForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lscp import LSCP
from pyod.models.mcd import MCD
from pyod.models.pca import PCA

# Load the dataset
data = pd.read_csv(r"C:\Users\smoha\Downloads\Wind_Turbine\CS795---NVDA-Emotion-Synth-main (2)\CARE_To_Compare_Data\CARE_To_Compare\Wind Farm A\Wind Farm A\datasets\0.csv")

# Select all features except time_stamp, asset_id, id, train_test, and status_type_id
features = [col for col in data.columns if col not in ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']]
X = data[features]

# Convert string values to numerical values
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


# Clustering algorithms
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
kmeans_labels = kmeans.labels_

agglomerative = AgglomerativeClustering(n_clusters=3)
agglomerative.fit(X_scaled)
agglomerative_labels = agglomerative.labels_

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
dbscan_labels = dbscan.labels_

meanshift = MeanShift()
meanshift.fit(X_scaled)
meanshift_labels = meanshift.labels_

gmm = GaussianMixture(n_components=3)
gmm.fit(X_scaled)
gmm_labels = gmm.predict(X_scaled)

# Anomaly detection algorithms
lof = LocalOutlierFactor(n_neighbors=20)
lof_scores = lof.fit_predict(X_scaled)

one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
one_class_svm.fit(X_scaled)
one_class_svm_scores = one_class_svm.score_samples(X_scaled)

elliptic_envelope = EllipticEnvelope(contamination=0.1)
elliptic_envelope.fit(X_scaled)
elliptic_envelope_scores = elliptic_envelope.decision_function(X_scaled)

isolation_forest = IsolationForest(contamination=0.1)
isolation_forest.fit(X_scaled)
isolation_forest_scores = isolation_forest.decision_function(X_scaled)

knn_outlier_detector = KNN(contamination=0.1)
knn_outlier_detector.fit(X_scaled)
knn_outlier_detector_scores = knn_outlier_detector.decision_function(X_scaled)

ocsvm = OCSVM(contamination=0.1)
ocsvm.fit(X_scaled)
ocsvm_scores = ocsvm.decision_function(X_scaled)

lof_pyod = LOF(contamination=0.1)
lof_pyod.fit(X_scaled)
lof_pyod_scores = lof_pyod.decision_function(X_scaled)

cblof = CBLOF(contamination=0.1)
cblof.fit(X_scaled)
cblof_scores = cblof.decision_function(X_scaled)

hbos = HBOS(contamination=0.1)
hbos.fit(X_scaled)
hbos_scores = hbos.decision_function(X_scaled)

abod = ABOD(contamination=0.1)
abod.fit(X_scaled)
abod_scores = abod.decision_function(X_scaled)

feature_bagging = FeatureBagging(contamination=0.1)
feature_bagging.fit(X_scaled)
feature_bagging_scores = feature_bagging.decision_function(X_scaled)

lscp = LSCP(contamination=0.1)
lscp.fit(X_scaled)
lscp_scores = lscp.decision_function(X_scaled)

mcd = MCD(contamination=0.1)
mcd.fit(X_scaled)
mcd_scores = mcd.decision_function(X_scaled)

pca = PCA(contamination=0.1)
pca.fit(X_scaled)
pca_scores = pca.decision_function(X_scaled)

# Print the results
print("Clustering Results:")
print("KMeans Labels:", kmeans_labels)
print("Agglomerative Clustering Labels:", agglomerative_labels)
print("DBSCAN Labels:", dbscan_labels)
print("Mean Shift Labels:", meanshift_labels)
print("Gaussian Mixture Model Labels:", gmm_labels)

print("\nAnomaly Detection Results:")
print("Local Outlier Factor Scores:", lof_scores)
print("One-Class SVM Scores:", one_class_svm_scores)
print("Elliptic Envelope Scores:", elliptic_envelope_scores)
print("Isolation Forest Scores:", isolation_forest_scores)
print("KNN Outlier Detector Scores:", knn_outlier_detector_scores)
print("OCSVM Scores:", ocsvm_scores)
print("LOF (PyOD) Scores:", lof_pyod_scores)
print("CBLOF Scores:", cblof_scores)
print("HBOS Scores:", hbos_scores)
print("ABOD Scores:", abod_scores)
print("Feature Bagging Scores:", feature_bagging_scores)
print("LSCP Scores:", lscp_scores)
print("MCD Scores:", mcd_scores)
print("PCA Scores:", pca_scores)
