import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


data = pd.read_csv('music_dataset_mod.csv')
data_copy = data.copy()
print(data_copy.head())
data_copy.info()
print(data_copy.isnull().sum())

data_cleaned = data_copy.dropna(subset=['Genre'])
print(data_cleaned.info())

unique_genres = data_cleaned['Genre'].unique()
print(f"Unique genres: {unique_genres}")
print(f"Number of unique genres: {len(unique_genres)}")

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(x='Genre', data=data_cleaned)
plt.title('Genre Distribution in Dataset')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

features = data_cleaned.drop(columns=['Genre'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

data_cleaned_scaled = pd.DataFrame(features_scaled, columns=features.columns)
data_cleaned_scaled['Genre'] = data_cleaned['Genre']

X=data_cleaned_scaled.drop(columns=['Genre'])
Y=data_cleaned_scaled['Genre']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)


new_data=X.copy()
new_data['Genre']=y_encoded

corr_matrix=new_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap de la matrice de corrélation des caractéristiques")
plt.show()


pca=PCA()
pca_data=pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_
print("Variance expliquée par chaque composant principal :")
print(explained_variance)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Nombre de Composants Principaux')
plt.ylabel('Variance Cumulative Expliquée')
plt.title('Variance Cumulative Expliquée par les Composants Principaux')
plt.axhline(y=0.80, color='r', linestyle='-')  # Tracer une ligne pour 80% de variance
plt.axvline(x=np.argmax(explained_variance >= 0.80) + 1, color='g', linestyle='--') 
plt.show()


pca = PCA(n_components=0.80)
features_pca = pca.fit_transform(X)
print(features_pca)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(features_pca, y_encoded, test_size=0.30, random_state=42)
logreg_pca = LogisticRegression(max_iter=10000, random_state=42,class_weight='balanced')
logreg_pca.fit(X_train_pca, y_train)
y_pred_pca = logreg_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("Accuracy on PCA-transformed data:", accuracy_pca)
print("Classification Report for PCA-transformed data:\n", classification_report(y_test, y_pred_pca))

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30, random_state=42)
logreg_scaled = LogisticRegression(max_iter=10000, random_state=42,class_weight='balanced')
logreg_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = logreg_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print("Accuracy on original scaled data:", accuracy_scaled)
print("Classification Report for original scaled data:\n", classification_report(y_test, y_pred_scaled))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_pca)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix for PCA-transformed Data")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Step 1: Isolate Unknown Genre Data
unknown_genre_data = data_copy[data_copy['Genre'].isnull()]
print("Unknown Genre Data:")
print(unknown_genre_data)
X_unknown = unknown_genre_data.drop(columns=['Genre'])
X_unknown_scaled = scaler.transform(X_unknown)
X_unknown_pca = pca.transform(X_unknown_scaled)
if accuracy_pca > accuracy_scaled:
    model = logreg_pca
    X_to_predict = X_unknown_pca
else:
    model = logreg_scaled
    X_to_predict = X_unknown_scaled
y_pred_unknown = model.predict(X_to_predict)
predicted_genres = label_encoder.inverse_transform(y_pred_unknown)
data_copy.loc[data_copy['Genre'].isnull(), 'Genre'] = predicted_genres

print("Updated DataFrame with Predicted Genres:")
print(data_copy.head())


# Calculate the correlation coefficient between 'Metal Frequencies' and 'Distorted Guitar'
correlation = data_copy['Metal Frequencies'].corr(data_copy['Distorted Guitar'])
print(f"Correlation coefficient between Metal Frequencies and Distorted Guitar: {correlation}")

pca = PCA()
pca.fit(X)
# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# Find the minimum number of components that capture at least 80% of the variance
num_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
# Output the result
print(f"Minimum number of components that capture at least 80% of the variance: {num_components_80}")

