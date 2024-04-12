# %% [markdown]
# K nearest Neighbours

# %% [markdown]
# Importing Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Importing Data Set

# %%
df=pd.read_csv("Social_Network_Ads.csv")
df

# %%
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x,y)


# %% [markdown]
# splitting The Data Set Into training and testing Sets

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)

# %%
print(x_train,x_test,y_train,y_test)

# %% [markdown]
# Features Scalling

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_test=sc.fit_transform(x_test)
x_train=sc.fit_transform(x_train)

# %%
print(x_train,x_test)

# %% [markdown]
# Training the model on the training set

# %%
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
classifier.fit(x_train,y_train)

# %% [markdown]
# Predicting New Result

# %%
print(classifier.predict(sc.transform([[30,87000]])))

# %%
y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape((len(y_pred)),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Confussion Metrix

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# %% [markdown]
# Plotting The Confussion Matrix 

# %%
import seaborn as sns
sns.heatmap(cm,fmt="g",annot=True)
plt.ylabel("Prediction")
plt.xlabel("Actual")
plt.title("Confusion Matrix")
plt.show()
plt.savefig("ConfussionMatrix.png")

# %% [markdown]
# Classifiction Scores 

# %%
print(classification_report(y_test,y_pred))

# %% [markdown]
# Visualizing The Training Set Results

# %%
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
plt.savefig("TrainingSet.png")

# %% [markdown]
# Visualizing The Testing Set Results

# %%
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
plt.savefig("TestSet.png")

# %% [markdown]
# Testing For Different Values of k

# %%
for i in range(1,20):
    classi=KNeighborsClassifier(n_neighbors=i,p=2,metric="minkowski")
    classi.fit(x_train,y_train)
    y_pred=classi.predict(x_test)
    score=accuracy_score(y_pred,y_test)
    print("K value = ",i)
    print("Confusion Matrix")
    print(confusion_matrix(y_test,y_pred))
    print("Classification Report")
    print(classification_report(y_test,y_pred))
    print("Accuracy Score= ",score)
    

# %%



