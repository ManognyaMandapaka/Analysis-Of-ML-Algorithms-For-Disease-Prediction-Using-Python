#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.model_selection import train_test_split
import csv,numpy as np,pandas as pd
import os


# In[1]:


data = pd.read_csv('data/Training1.csv')


# In[4]:


data.head()


# In[5]:


data['prognosis'].value_counts().plot.bar(title='Classification Frequency')


# In[6]:


df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]


# In[7]:


x = df[cols]
y = df['prognosis']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, train_size = 0.25, random_state=42)


# In[9]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# # Support Vector Machine

# In[10]:


from sklearn.svm import SVC
SVM = SVC(kernel='linear')
SVM.fit(X_train, y_train)
predictions = SVM.predict(X_test)
val1 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for SVM: ", val1, "\n")
print("*Confusion Matrix for SVM: ")
print(confusion_matrix(y_test, predictions))
print("*Classification Report for SVM: ")
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)
print(accuracy_score(y_test, predictions))

plt.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()


# # Decision Tree

# In[11]:


from sklearn import tree
DT = tree.DecisionTreeClassifier()
DT.fit(X_train, y_train)
predictions = DT.predict(X_test)
val2 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for DT: ", val2, "\n")
print("*Confusion Matrix for DT: ")
print(confusion_matrix(y_test, predictions))
print("*Classification Report for DT: ")
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)
print(accuracy_score(y_test, predictions))

plt.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()


# # Random Forest

# In[12]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
predictions = RF.predict(X_test)
val3 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for RF: ", val3, "\n")
print("*Confusion Matrix for RF: ")
print(confusion_matrix(y_test, predictions))
print("*Classification Report for RF: ")
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)
print(accuracy_score(y_test, predictions))

plt.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()


# # Naive Bayes

# In[13]:


from sklearn import naive_bayes
# Instanciate the classifier
GNB = naive_bayes.GaussianNB()
GNB.fit(X_train, y_train)
predictions = GNB.predict(X_test)
val4 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for GNB: ", val4, "\n")
print("*Confusion Matrix for GNB: ")
print(confusion_matrix(y_test, predictions))
print("*Classification Report for GNB: ")
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

plt.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()


# # MLP

# In[14]:


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier()
MLP.fit(X_train, y_train)
predictions = MLP.predict(X_test)
val5 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for MLP: ", val5, "\n")
print("*Confusion Matrix for MLP: ")
print(confusion_matrix(y_test, predictions))
print("*Classification Report for MLP: ")
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

plt.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()


# # KNN

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, y_train)
predictions = KNN.predict(X_test)
val6 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for KNN: ", val6, "\n")
print("*Confusion Matrix for KNN: ")
print(confusion_matrix(y_test, predictions))
print("*Classification Report for KNN: ")
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

plt.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()


# # Model Comparision

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bring some raw data.
frequencies = [val1,val2,val3,val4,val5,val6]

# In my original code I create a series and run on that,
# so for consistency I create a series from the list.
freq_series = pd.Series(frequencies)

x_labels = ['SVM', 'DT','RF','GNB','MLP','KNN']

# Plot the figure.
plt.figure(figsize=(12, 8))
ax = freq_series.plot(kind='bar')
ax.set_title('Evaluation of ML & DL')
ax.set_xlabel('Classifier!')
ax.set_ylabel('Accuracy Range')
ax.set_xticklabels(x_labels)


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.4f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# Call the function above. All the magic happens there.
add_value_labels(ax)
plt.show()
#plt.savefig("image.png")


# # Application

# In[3]:


from werkzeug.wrappers import Request, Response
import csv
from flask import Flask, render_template,request,redirect,url_for
import diseaseprediction


# In[ ]:


app = Flask(__name__)
with open('templates/Testing.csv', newline='') as f:
    reader = csv.reader(f)
    symptoms = next(reader)
    symptoms = symptoms[:len(symptoms)-1]


# In[ ]:


@app.route('/', methods=['GET'])
def dropdown():
        return render_template('includes/default.html', symptoms=symptoms)

@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    selected_symptoms = []
    if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom1'])
    if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom2'])
    if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom3'])
    if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom4'])
    if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom5'])

    # disease_list = []
    # for i in range(7):
    #     disease = diseaseprediction.dosomething(selected_symptoms)
    #     disease_list.append(disease)
    # return render_template('disease_predict.html',disease_list=disease_list)
    disease = diseaseprediction.dosomething(selected_symptoms)
    return render_template('disease_predict.html',disease=disease,symptoms=symptoms)

# @app.route('/default')
# def default():
#         return render_template('includes/default.html')
 
@app.route('/find_doctor', methods=['POST'])
def get_location():
    location = request.form['doctor']
    return render_template('find_doctor.html',location=location,symptoms=symptoms)

@app.route('/drug', methods=['POST'])
def drugs():
    medicine = request.form['medicine']
    return render_template('home.html',medicine=medicine,symptoms=symptoms)


# In[ ]:


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)


# In[ ]:




