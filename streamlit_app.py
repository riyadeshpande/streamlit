import pandas as pd
import numpy as np
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
#from hyperopt import fmin, tpe, hp
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import tensorflow as tf
from tensorflow import keras

def makelabels(labels):
    i = 0
    while i < len(labels):
        if labels[i] == 0:
            labels[i] = "Accepted"
        else:
            labels[i] = "Eliminated"
        i = i + 1
    return labels

def remcols(table,indices):
    for idx in indices:
        table = table[table[idx] != "Not Recorded"]
    return table


M03 = pd.read_excel("./dataset/M03.xlsx")
M06 = pd.read_excel("./dataset/M06.xlsx")
M09 = pd.read_excel("./dataset/M09.xlsx")
M12 = pd.read_excel("./dataset/M12.xlsx")


Env_Idx        = ["BehElim", "Confidence", "Concentration", "Responsiveness", "Initiative",
                  "Excitability", "Hearing Sensitivity", "Body Sensitivity", "CR", "IP", "PP"]
Test_Idx       = ["BehElim", "CR.1", "MP", "PP.1", "IP.1", "HG", "H1", "H2", "ACT"]
Env_Idx1       = ["Confidence", "Concentration", "Responsiveness", "Initiative",
                  "Excitability", "Hearing Sensitivity", "Body Sensitivity", "CR", "IP", "PP"]
Test_Idx1      = ["CR.1", "MP", "PP.1", "IP.1", "HG", "H1", "H2", "ACT"]


# Remove dogs from the tests if they contain missing information
M03_Env       = remcols(M03[Env_Idx].dropna(), Env_Idx1)
M03_Test      = remcols(M03[Test_Idx].dropna(), Test_Idx1)
M06_Env       = remcols(M06[Env_Idx].dropna(), Env_Idx1)
M06_Test      = remcols(M06[Test_Idx].dropna(), Test_Idx1)
M09_Env       = remcols(M09[Env_Idx].dropna(), Env_Idx1)
M09_Test      = remcols(M09[Test_Idx].dropna(), Test_Idx1)
M12_Env       = remcols(M12[Env_Idx].dropna(), Env_Idx1)
M12_Test      = remcols(M12[Test_Idx].dropna(), Test_Idx1)


X = M12_Env
y           = X["BehElim"]
X           = X.drop("BehElim", axis = 1)
beh_labels  = makelabels(list(y))

X['Hearing Sensitivity'] = X['Hearing Sensitivity'].astype('float64')
X['Body Sensitivity'] = X['Body Sensitivity'].astype('float64')
X['CR'] = X['CR'].astype('float64')
X['IP'] = X['IP'].astype('float64')
X['PP'] = X['PP'].astype('float64')

# Make Training/Testing (70/30%) Datasets for Single Tests
X1_train, X1_test, y1_train, y1_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.30,
                                                        random_state = 101)


#GRADIENT BOOSTING MACHINE
def gradient_boosting_model(X_train, y_train, X_test, y_test):
    best = {'learning_rate': 0.13787546683029264,
    'max_depth': 2.0,
    'min_samples_leaf': 0.14336262263632524,
    'min_samples_split': 0.47970789631686717,
    'n_estimators': 137.0}
    gb_model = GradientBoostingClassifier(
        learning_rate=best['learning_rate'],
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        min_samples_split=best['min_samples_split'],
        min_samples_leaf=best['min_samples_leaf']
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    return gb_model, round(accuracy_score(gb_pred, y_test), 2), round(roc_auc_score(y_test, gb_pred), 3), round(f1_score(y_test, gb_pred, average=None)[0], 2), round(f1_score(y_test, gb_pred, average=None)[1], 2)

gb_model, a,b,c,d = gradient_boosting_model(X1_train, y1_train, X1_test, y1_test)


#LOGISTIC REGRESSION
def lr_model(X1_train, y1_train, X1_test, y1_test):
  log_model1  = LogisticRegression()
  log_model1.fit(X1_train,y1_train)
  log_pred1   = log_model1.predict(X1_test)
  return log_model1, round(accuracy_score(y1_test, log_pred1),2), round(roc_auc_score(y1_test, log_model1.predict(X1_test)), 3), round(f1_score(y1_test, log_pred1, average=None)[0],2), round(f1_score(y1_test, log_pred1, average=None)[1],2)

LogisticRegression_model, a,b,c,d = lr_model(X1_train, y1_train, X1_test, y1_test)


#SUPPORT VECTOR MACHINE
def svm_model(X1_train, y1_train, X1_test, y1_test):
  svc_model1  = SVC()
  svc_model1.fit(X1_train,y1_train)
  svc_pred1   = svc_model1.predict(X1_test)
  return svc_model1, round(accuracy_score(svc_pred1, y1_test),2), round(roc_auc_score(y1_test, svc_pred1),3), round(f1_score(y1_test, svc_pred1, average=None)[0],2), round(f1_score(y1_test, svc_pred1, average=None)[1],2)

SupportVector_model, a,b,c,d = svm_model(X1_train, y1_train, X1_test, y1_test)


#RANDOM FOREST
def rf_model(X1_train, y1_train, X1_test, y1_test):
  rfc_model1  = RandomForestClassifier(n_estimators = 200, max_depth=20)
  rfc_model1.fit(X1_train, y1_train)
  rfc_pred1   = rfc_model1.predict(X1_test)
  return rfc_model1, round(accuracy_score(rfc_pred1, y1_test),2), round(roc_auc_score(y1_test, rfc_pred1),3), round(f1_score(y1_test, rfc_pred1, average=None)[0],2), round(f1_score(y1_test, rfc_pred1, average=None)[1],2)

RandomForest_model, a,b,c,d = rf_model(X1_train, y1_train, X1_test, y1_test)


#DECISION TREE
def dtree_model(X1_train, y1_train, X1_test, y1_test):
  s = ['gini', 'entropy']
  best = {'criterion': 1,
  'max_depth': 9.0,
  'min_samples_leaf': 0.29895651088400965,
  'min_samples_split': 0.9661902063483865}
  d_model1  = DecisionTreeClassifier(criterion=s[best['criterion']], max_depth=int(best['max_depth']), min_samples_split=best['min_samples_split'], min_samples_leaf=best['min_samples_leaf'])
  d_model1.fit(X1_train, y1_train)
  d_pred1   = d_model1.predict(X1_test)
  return d_model1, round(accuracy_score(d_pred1, y1_test),2), round(roc_auc_score(y1_test, d_pred1),3), round(f1_score(y1_test, d_pred1, average=None)[0],2), round(f1_score(y1_test, d_pred1, average=None)[1],2)

DecisionTree_model, a,b,c,d = dtree_model(X1_train, y1_train, X1_test, y1_test)


#EXTREME GRADIENT BOOSTING
def xgboost_model(X_train, y_train, X_test, y_test):
    best = {'gamma': 0.4862373249390205,
    'learning_rate': 0.29024715071124096,
    'max_depth': 4.0,
    'min_child_weight': 6.232215063492531,
    'n_estimators': 62.0,
    'subsample': 0.9500106895253753}
    xgb_model = xgb.XGBClassifier(
        learning_rate=best['learning_rate'],
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        min_child_weight=best['min_child_weight'],
        subsample=best['subsample'],
        gamma=best['gamma']
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    return xgb_model, round(accuracy_score(xgb_pred, y_test), 2), round(roc_auc_score(y_test, xgb_pred), 3), round(f1_score(y_test, xgb_pred, average=None)[0], 2), round(f1_score(y_test, xgb_pred, average=None)[1], 2)

XGB_model, a,b,c,d = xgboost_model(X1_train, y1_train, X1_test, y1_test)


#ARTIFICAL NEURAL NETWORK
def neural_network_model(X_train, y_train, X_test, y_test):
  X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
  Y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int64)

  X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
  Y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int64)

  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1],)),  # Input layer (flatten the input data)
    keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    keras.layers.Dropout(0.5),  # Add dropout with a 50% dropout rate
    keras.layers.Dense(64, activation='relu'),   # Another hidden layer with 64 neurons and ReLU activation
    keras.layers.Dense(2, activation='softmax')  # Output layer with 10 neurons for classification
])

  # Compile the model
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',  # For classification tasks with integer labels
                metrics=['accuracy'])

  # Train the model
  model.fit(X_train_tf, Y_train_tf, epochs=25, batch_size=32)  # You can adjust the number of epochs and batch size
  nn_pred = model.predict(X_test_tf)
  nn_pred1 = np.zeros(X_test.shape[0])
  for i in range(X_test.shape[0]):
    nn_pred1[i] = 0 if nn_pred[i][0]>nn_pred[i][1] else 1
  test_loss, test_accuracy = model.evaluate(X_test_tf, Y_test_tf)
  return model, round(test_accuracy, 2), round(roc_auc_score(Y_test_tf, nn_pred1), 3), round(f1_score(Y_test_tf, nn_pred1, average=None)[0], 2), round(f1_score(Y_test_tf, nn_pred1, average=None)[1], 2)

#ANN_model, a,b,c,d = neural_network_model(X1_train, y1_train, X1_test, y1_test)


def adaboost_model(X_train, y_train, X_test, y_test):
    best = {'learning_rate': 0.05085118741543816, 'n_estimators': 157.0}
    adaboost_model = AdaBoostClassifier(
        n_estimators=int(best['n_estimators']),
        learning_rate=best['learning_rate']
    )
    adaboost_model.fit(X_train, y_train)
    adaboost_pred = adaboost_model.predict(X_test)
    return adaboost_model, round(accuracy_score(adaboost_pred, y_test), 2), round(roc_auc_score(y_test, adaboost_pred), 3), round(f1_score(y_test, adaboost_pred, average=None)[0], 2), round(f1_score(y_test, adaboost_pred, average=None)[1], 2)

Ada_model, a,b,c,d = adaboost_model(X1_train, y1_train, X1_test, y1_test)


#K NEAREST NEIGHBOURS
def knn_classifier_model(X_train, y_train, X_test, y_test):
    w = ['uniform', 'distance']
    p_val = [1,2]
    best = {'n_neighbors': 6.0, 'p': 1, 'weights': 1}
    knn_model = KNeighborsClassifier(
        n_neighbors=int(best['n_neighbors']),
        weights=w[best['weights']],
        p=p_val[best['p']]
    )
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    return knn_model, round(accuracy_score(knn_pred, y_test), 2), round(roc_auc_score(y_test, knn_pred), 3), round(f1_score(y_test, knn_pred, average=None)[0], 2), round(f1_score(y_test, knn_pred, average=None)[1], 2)

KNN_model, a,b,c,d = knn_classifier_model(X1_train, y1_train, X1_test, y1_test)


#RIDGE CLASSIFIER
def ridge_classifier_model(X_train, y_train, X_test, y_test):
    best = {'alpha': 0.07436964667090645, 'solver': 6}
    s = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    ridge_model = RidgeClassifier(
        alpha=best['alpha'],
        solver=s[best['solver']]
    )
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    return ridge_model, round(accuracy_score(ridge_pred, y_test), 2), round(roc_auc_score(y_test, ridge_pred), 3), round(f1_score(y_test, ridge_pred, average=None)[0], 2), round(f1_score(y_test, ridge_pred, average=None)[1], 2)

Ridge_model, a,b,c,d = ridge_classifier_model(X1_train, y1_train, X1_test, y1_test)


#QUADRATIC DISCRIMINANT ANALYSIS
def qda_classifier_model(X_train, y_train, X_test, y_test):
    best = {'reg_param': 0.25177877773076285}
    qda_model = QuadraticDiscriminantAnalysis(
        reg_param=best['reg_param']
    )
    qda_model.fit(X_train, y_train)
    qda_pred = qda_model.predict(X_test)
    return qda_model, round(accuracy_score(qda_pred, y_test), 2), round(roc_auc_score(y_test, qda_pred), 3), round(f1_score(y_test, qda_pred, average=None)[0], 2), round(f1_score(y_test, qda_pred, average=None)[1], 2)

QDA_model, a,b,c,d = qda_classifier_model(X1_train, y1_train, X1_test, y1_test)


#LINEAR DISCRIMINANT ANALYSIS
def lda_classifier_model(X_train, y_train, X_test, y_test):
    best = {'shrinkage': 0.5323495407563335, 'solver': 0}
    s = ['lsqr', 'eigen']
    lda_model = LinearDiscriminantAnalysis(
        solver=s[best['solver']],
        shrinkage=best['shrinkage']
    )
    lda_model.fit(X_train, y_train)
    lda_pred = lda_model.predict(X_test)
    return lda_model, round(accuracy_score(lda_pred, y_test), 2), round(roc_auc_score(y_test, lda_pred), 3), round(f1_score(y_test, lda_pred, average=None)[0], 2), round(f1_score(y_test, lda_pred, average=None)[1], 2)

LDA_model, a,b,c,d = lda_classifier_model(X1_train, y1_train, X1_test, y1_test)


model_names = ['Gradient Boosting', 'Logistic Regression', 'Support Vector Machine', 'Random Forest', 'Decision Tree', 'XGBoost', 'ANN', 'Adaboost', 'KNN', 'Ridge Classifier', 'QDA', 'LDA']
models = [
    # Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())]),
    # Pipeline([('model', MLPClassifier())]),
    Pipeline([('model', gb_model)]),
    Pipeline([('model', LogisticRegression_model)]),
    Pipeline([('model', SupportVector_model)]),
    Pipeline([('model', RandomForest_model)]),
    Pipeline([('model', DecisionTree_model)]),
    Pipeline([('model', XGB_model)]),
    Pipeline([('model', Ada_model)]),
    Pipeline([('model', KNN_model)]),
    Pipeline([('model', Ridge_model)]),
    Pipeline([('model', QDA_model)]),
    Pipeline([('model', LDA_model)]),
]

# Streamlit App

st.title("Canine Behavioural Classification")
st.caption("Canines possess a sense of smell roughly forty times more sensitive than human's and because of their acute sense of smell, they are used as working dogs in various fields. ")

col1, col2, col3 = st.columns([1,6,1])
with col1:
  st.write("")
with col2:
  st.image("./images/airport_dog.png", width = 400)
  st.image("./images/airport_doggo.png", width = 400)
with col3:
  st.write("")
st.caption("Working dogs are trained to detect a wide variety of specific odors, such as explosives in crowded environments like concert halls or airports. Transportation Security Administration (TSA) trains around 400 canines every year to screen luggage at mass transits for narcotics, counterfeit money and even lithium batteries. Dogs are taught by their handlers to alert to the presence of these odors with a learned response such as sitting or lying quietly. Canine olfactory capabilities are more reliable than x-ray or trace machines in security.")


col1, col2, col3 = st.columns([1,6,1])
with col1:
  st.write("")
with col2:
  st.image("./images/police_dog.jpg", width = 400)
  st.image("./images/quake_dog.png", width = 400)
with col3:
  st.write("")
st.caption("K-9 dogs (also known as polic dogs) assist law enforcement officers by locating missing persons, finding crime scene evidence and attacking suspects who flee from officers. Dogs can search through rubble after an earthquake, or cover miles of forest looking for a lost hiker and even locate the bodies of drowned victims in oceans.")

col1, col2, col3 = st.columns([1,6,1])
with col1:
  st.write("")
with col2:
  st.image("./images/military_dogs.png", width = 400)
with col3:
  st.write("")
st.caption("Many military working dogs (MWDs) are known to detect landmines in order to protect personnel from danger. Depending on their individual aptitudes, war dogs are trained to attack, hold down, and incapacitate the enemy. Top military breeds like German Shepherds or Belgian Malinois are particularly athletic and are selected by trainers for their aggressive natures.")
st.caption("Dogs have also dragged the wounded to safety or retreived medical help by barking. Fighter dogs chase down and restrain fugitives and prisoners. PTSD service dogs help veterans feel safe in their homes and have even brought many back from thoughts of suicide.")
st.caption("Scout dogs are trained to smell and listen for threats located as far as 1,000 feet away and even through dark tunnels. They have reportedly sensed the presence of weapon cashes, ambushes, and enemy platoons hiding underwater, saving many lives.")

col1, col2, col3 = st.columns([1,6,1])
with col1:
  st.write("")
with col2:
  st.image("./images/service_dog_in_training.jpg", width = 400)
  st.image("./images/service_dog.jpg", width = 400)
with col3:
  st.write("")
st.caption("Dogs can give warning signs to patients about to have a migraine, detect the onset of a seizure as well as impending cardiac episodes. Dogs have also shown they can detect cancer by sniffing people's skin, bodily fluids, or breath. They also provide emotional support to individuals with depression by reminding their owner to take medication, and interrupting self-harming behaviors.")


col1, col2, col3 = st.columns([1,6,1])
with col1:
  st.write("")
with col2:
  st.image("./images/covid_dog.png", width = 400)
  st.image("./images/helsinki_airport_dog.png", width = 400)
with col3:
  st.write("")
st.caption("Dubai Airport was the first to employ dogs to sniff out Covid-19 in traveler's sweat. Finland and Miami airports also employed sniffer dogs for the same thereafter. This process is not only cheap but also highly efficient in detecting the virus.")


col1, col2, col3 = st.columns([1,8,1])
with col1:
  st.write("")
with col2:
  st.video("./images/medication.mp4")
with col3:
  st.write("")
st.caption("They perform essential actions like guiding people with visual impairments in day to day activities such as recognising obstacles in their path and using public transport safely. For deaf people, guide dogs work to alert them to sounds of the doorbell, telephone alerts and even smoke alarms.")

col1, col2, col3 = st.columns([1,6,1])
with col1:
  st.write("")
with col2:
  st.image("./images/service_dog_blind.jpg", width = 500)
with col3:
  st.write("")
st.caption("They can give warning signs to patients about to have a migraine by various gestures and direct them to lie or sit down in order to prepare for the attack. Dogs have notified their owners suffering from diabetes when their blood sugar had dropped or spiked to a life threatening level.")
st.caption("Therapists involve dogs to help children with learning disabilities to improve their memory and those with sensory processing challenges to work on their fine motor skills.")

st.subheader("What is TSA?")
st.image("./images/tsa_dog.jpg", width = 500)
st.caption("The Transportation Security Administration (TSA) is an agency of the United States Department of Homeland Security (DHS) that is in charge of the country's transportation networks and infrastructure.")
st.caption("The TSA's primary mission is airport security and the prevention of aircraft hijacking. It is responsible for screening passengers and baggage at more than 450 U.S. airports, employing screening officers, explosives detection dog handlers, and bomb technicians in airports. The TSA has screening processes and regulations related to passengers and checked and carry-on luggage, including identification verification, pat-downs, full-body scanners, and explosives screening.")
st.image("./images/tsa_airport_dog.jpeg", width = 500)
st.caption("The TSA National Explosives Detection Canine Program oversees the canine teams that do the job of protecting the transportation domain. They are trained on a variety of explosives and graduate the course after exhibiting proficiency in venues inclusive of all transportation environments including airport, terminal, freight, cargo, baggage, vehicle, bus, ferry and rail. There are 17 indoor venues on the premises that mimic a variety of transportation sites and modes. In relation to airports, this includes a cargo facility, an airport gate area, a checkpoint, a baggage claim area, the interior of an aircraft, an air cargo facility, two mock terminals, and open area searches venues for air scenting. Kennels can accommodate approximately 350 dogs.")
st.caption("Once a team graduates from the training program, they return to their duty station to acclimate and familiarize the canine to their assigned operational environment. Each team is continually assessed to ensure the canines demonstrate operational proficiency in their environment including four key elements: the canine's ability to recognize explosives odors, the handler's ability to interpret the canine's change of behavior, the handler's ability to conduct logical and systematic searches, and the team's ability to locate the explosives odor source. Canine teams work at more than 100 of the nation's airports, mass-transit and maritime systems and are a highly mobile and efficient explosives detection tool. ")

st.subheader("Dataset")
st.caption("The Labrador and Golden Retriever are the preferred breeds for a guide dog throughout the world. They are highly trainable placid dogs that are responsive, quick to learn, sensitive, and possess temperaments required to thrive as service dogs.")
st.caption("The dataset used in this study is taken from the cohort of Transportation Security Administration (TSA) canine breeding and training program where 628 Labrador Retrievers were evaluated based on a series of tests taken over a 12-month period. The dogs were assessed based on their performance in these tests related to olfaction detection traits, cooperation with handler and general activity. On basis of their scores by the end of this period, they were either successfully accepted or eliminated from the training program for working dogs.")
st.image("./images/dataset_m12.png", width = 800)
st.caption("The dataset used in the present work is collected from dogs fostered in the period from 2002-2013. Beginning at the age of three months, they were evaluated at four time-points, with each evaluation taking place three months after the last. At each time-point, the dogs were made to go through two evaluation tests - Airport Terminal test and Environmental test. The former test was done in a mock airport terminal with controlled setting wherein the dogs had to hunt for a scented towel in vessels scattered throughout the terminal. This test was done to measure the ability to focus on an object even after being hidden, concentration, willingness and level of engagement with the handler. Whereas, the latter test was done in various locations at each time-point. These included a busy gift shop, a woodshop, a cargo area and airport passenger locations. This test was performed to capture the traits related to reactivity to noise stimulus, lack of distraction and enthusiasm.")
st.caption("The dataset consists of four files namely M03, M06, M09 and M012 taken at the four time-points in the fostering period. Each file consists of the evaluation results from both the tests. There are total 33 columns which include the name, litter ID, test location, the different traits measured for behaviour pattern, comments from the handler and the final column gives the binary number indicating whether the dog is accepted further for pre-training or eliminated due to behavioural reasons.")


st.subheader("Motivation")
st.caption("Most canines spend between 6-12 months in a training program to understand obedience, follow instructions and practice their skill in various environments.")
st.caption("However, the training programs required for these dogs are expensive and bear a low acceptance rate. As of May 2021, in TSA canine programs, the average cost to train a canine a traditional explosives detection canine and handler is \$33,000 and that for a passenger screening canine and handler is \$46,000, which is over INR 38 lakhs. Therefore, there is a need for a more productive way to operate these programs with the existing standard datasets available in public domain.")
st.caption("Eyre et. al. have implemented three supervised machine learning algorithms but have noted the need for improvement in accuracy and also for addressing the issue of class imbalance. The minority class comprises of only 15% of the entire dataset causing the models to be biased towards the majority class.")

st.subheader("Aim of the project")
st.write("The aim of this paper is to demonstrate the use of ML to improve the selection and training process with a high degree of accuracy.")
st.caption("In this present work, we have applied thirteen supervised machine learning algorithms to classify and predict whether a canine is more likely to get accepted or eliminated based on its behavioural traits. We have addressed the problem of class imbalance by using SMOTE (Synthetic Minority Oversampling Technique) that creates artificial samples of the minority class based on the interpolation between instances of the feature space, rather than replicating the already existing ones. This helps generate new synthetic data points for the minority class, thereby overcoming the shortage and boosting the model's performance on the minority class predictions. HyperOpt was used to fine-tune the hyperparamters so that the best possible accuracy could be achieved.")

st.header("This webapp allows you to set custom values of features and assess canine suitability for olfactory detection training program.")
st.markdown("Lets try it!")

st.image("./images/group_of_dogs.png")

# Sidebar
selected_model = st.sidebar.selectbox("Select a Model", model_names)

# Display Model Information
st.subheader(f"Model : {selected_model}")
st.write("Here's a demo for the selected ML model.")
st.write("You may input trait values below to make predictions.")

# User Input
user_input = {}
for feature in X.columns:
    #user_input[feature] = st.slider(f"Enter {feature}:", min_value=X[feature].min().astype('float64'), max_value=X[feature].max().astype('float64'), value=X[feature].mean())
    user_input[feature] = st.slider(f"Enter {feature}:", min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].mean()))
# Make Prediction
model_index = model_names.index(selected_model)
selected_model_obj = models[model_index]

label = ['Accepted', 'Eliminated']
if st.button("Make Prediction"):
    user_data = pd.DataFrame([user_input])
    prediction = selected_model_obj.predict(user_data)
    if prediction[0]==0:
      st.success(f"The model predicts: {label[prediction[0]]}")
    else:
      st.error(f"The model predicts: {label[prediction[0]]}")

st.image("./images/model_comp_imbalance.png")
st.caption("Comparison of accuracy and AUC of the ML models")

st.image("./images/gbm_metrics.png")
st.caption("Performance metrics of GBM on the combined dataset")
st.write("The best performance was obtained by GBM on M36912 combined dataset. We have achieved a remarkable accuracy of 96% along with AUC of 0.81 compared to Eyre's accuracy of 88% and AUC of 0.61.")

st.image("./images/comparison.png")
st.caption("Best performing models on M03 AT test")

st.image("./images/acceptance.png")
st.subheader("Our abstract has been accepted in Biosangam 2024 International conference organised by the Department of Biotechnology at MNNIT Allahabad to be held from 23rd to 25th February 2024. This is the sixth edition of the conference and is mainly focused on Bio-Technological Intervention for Health, Agriculture and Circular Economy.")
