import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


data=pd.read_csv("C:\\Users\\bulip\\Documents\\Hackathon-2023\\food-data.csv")

data['wastage%']=0
for i in range(len(data)): #iterating between all the rows of dataframe
    data['wastage%'][i] = ((data['served-qty'][i] - data['consumed-qty'][i])/data['served-qty'][i])*100

data['consumption%']=0
for i in range(len(data)): #iterating between all the rows of dataframe
    data['consumption%'][i] = ((data['consumed-qty'][i])/data['served-qty'][i])*100

data.replace({'day':{'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5}},inplace=True) 


data.replace({'event':{'breakfast':1,'lunch':2,'snacks':3}},inplace=True) 
data.replace({'menu':{'idli':1, 'boiled egg':2, 'Chicken curry':3, 'veg biriyani':4, 'Chappati':5,'Veg meals':6, 'Kheer':7, 'Samosa':8, 'rawa idli':9, 'Jeera rice':10,'Rosgulla':11, 'Curry bun':12, 'poori':13, ' Chicken Biryani':14, 'Veg pulav':15,'Gulab jamun':16, 'egg Puff':17, 'Puff':18, 'Vada':19, 'Puliogere':20,'Fruit salad':21, 'thatte idli':22, 'Chicken biriyani':23, 'Veg fried rice':24,'masala dosa':25}},inplace=True) 
data.replace({'diet':{'Vegeterian':1,'Non-Vegeterian':2}},inplace=True) 

X = data.drop(['date','menu-id','flavor_profile','calorielevel-per-100gm','consumed-qty'],axis=1)
Y = data['consumed-qty'] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)
training_data_prediction = lin_reg_model.predict(X_train)
train_error_score = metrics.r2_score(Y_train, training_data_prediction) 
print("<br/><b>R squared Error - Training : ", train_error_score, ) 

Y_pred = lin_reg_model.predict(X_test)
test_error_score = metrics.r2_score(Y_test, Y_pred)

print("<br/><b>R squared Error - Test: ", test_error_score) 


monday=data[(data["day"] == 1)]
monday['iswasted'] = monday['wastage%'].apply(lambda x: 1 if x >= 20 else 0)
X = monday.drop(['date','menu-id','flavor_profile','calorielevel-per-100gm','iswasted'],axis=1)
y = monday['iswasted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("<br/><b>Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("<br/><b>recision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("<br/>Recall:",metrics.recall_score(y_test, y_pred))
sns.set(style='darkgrid')
fig = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={"color": "green"}, line_kws={"color": "blue"})

plt.savefig('predict-image.png')

idli=data[(data["day"] == 1) & (data["menu"] == 1) ]
boiledegg = data[(data["day"] == 1) & (data["menu"] == 2) ]
Chicken_curry = data[(data["day"] == 1) & (data["menu"] == 3) ]
veg_biriyani = data[(data["day"] == 1) & (data["menu"] == 4) ]
chap=data[(data["day"] == 1) & (data["menu"] == 5) ]
veg_meals = data[(data["day"] == 1) & (data["menu"] == 6) ]
desert= data[(data["day"] == 1) & (data["menu"] == 7) ]
samosa = data[(data['day'] == 1) & (data['menu'] == 8)]


def idli_wastage_pred():
    Xidli = idli.drop(['date', 'menu-id', 'flavor_profile',
                      'calorielevel-per-100gm', 'wastage%'], axis=1)
    Yidli = idli['wastage%']

    (Xidli_train, Xidli_test, Yidli_train, Yidli_test) = \
        train_test_split(Xidli, Yidli, test_size=0.2, random_state=42)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(Xidli_train, Yidli_train)
    training_idli_prediction = lin_reg_model.predict(Xidli_train)
    train_error_score = metrics.r2_score(Yidli_train,
            training_idli_prediction)

    Yidli_pred = lin_reg_model.predict(Xidli_test)
    test_error_score = metrics.r2_score(Yidli_test, Yidli_pred)

    print ("<br/><b>Predicted Idli wastage :", Yidli_pred)
    
    



def samosa_wastage_pred():
    Xsamosa = samosa.drop(['date', 'menu-id', 'flavor_profile',
                          'calorielevel-per-100gm', 'wastage%'], axis=1)
    Ysamosa = samosa['wastage%']

    (Xsamosa_train, Xsamosa_test, Ysamosa_train, Ysamosa_test) = \
        train_test_split(Xsamosa, Ysamosa, test_size=0.2,
                         random_state=42)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(Xsamosa_train, Ysamosa_train)
    training_samosa_prediction = lin_reg_model.predict(Xsamosa_train)
    train_error_score = metrics.r2_score(Ysamosa_train,
            training_samosa_prediction)
    
    Ysamosa_pred = lin_reg_model.predict(Xsamosa_test)
    test_error_score = metrics.r2_score(Ysamosa_test, Ysamosa_pred)
    
    print ("<br/><b>Predicted Samosa wastage :", Ysamosa_pred)
    


def roti_wastage_pred():
    Xchap = chap.drop(['date', 'menu-id', 'flavor_profile',
                      'calorielevel-per-100gm', 'wastage%'], axis=1)
    Ychap = chap['wastage%']

    (Xchap_train, Xchap_test, Ychap_train, Ychap_test) = \
        train_test_split(Xchap, Ychap, test_size=0.2, random_state=42)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(Xchap_train, Ychap_train)
    training_chap_prediction = lin_reg_model.predict(Xchap_train)
    train_error_score = metrics.r2_score(Ychap_train,
            training_chap_prediction)
    
    Ychap_pred = lin_reg_model.predict(Xchap_test)
    test_error_score = metrics.r2_score(Ychap_test, Ychap_pred)
   
    print ("<br/><b>Predicted Roti wastage :", Ychap_pred)
    

def boiledegg_wastage_pred():
    Xboiledegg = boiledegg.drop(['date', 'menu-id', 'flavor_profile',
                      'calorielevel-per-100gm', 'wastage%'], axis=1)
    Yboiledegg = boiledegg['wastage%']

    (Xboiledegg_train, Xboiledegg_test, Yboiledegg_train, Yboiledegg_test) = \
        train_test_split(Xboiledegg, Yboiledegg, test_size=0.2, random_state=42)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(Xboiledegg_train, Yboiledegg_train)
    training_boiledegg_prediction = lin_reg_model.predict(Xboiledegg_train)
    train_error_score = metrics.r2_score(Yboiledegg_train,
            training_boiledegg_prediction)
    Yboiledegg_pred = lin_reg_model.predict(Xboiledegg_test)
    test_error_score = metrics.r2_score(Yboiledegg_test, Yboiledegg_pred)
    print ("<br/><b>Predicted boiledegg wastage :", Yboiledegg_pred)

    


def Chicken_curry_wastage_pred():
    XChicken_curry = Chicken_curry.drop(['date', 'menu-id', 'flavor_profile',
                      'calorielevel-per-100gm', 'wastage%'], axis=1)
    YChicken_curry = Chicken_curry['wastage%']

    (XChicken_curry_train, XChicken_curry_test, YChicken_curry_train, YChicken_curry_test) = \
        train_test_split(XChicken_curry, YChicken_curry, test_size=0.2, random_state=42)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(XChicken_curry_train, YChicken_curry_train)
    training_Chicken_curry_prediction = lin_reg_model.predict(XChicken_curry_train)
    train_error_score = metrics.r2_score(YChicken_curry_train,
            training_Chicken_curry_prediction)
    YChicken_curry_pred = lin_reg_model.predict(XChicken_curry_test)
    test_error_score = metrics.r2_score(YChicken_curry_test, YChicken_curry_pred)
    print ("<br/><b>Predicted Chicken curry wastage :", YChicken_curry_pred)


def veg_biriyani_wastage_pred():
    Xveg_biriyani = veg_biriyani.drop(['date', 'menu-id', 'flavor_profile',
                      'calorielevel-per-100gm', 'wastage%'], axis=1)
    Yveg_biriyani = veg_biriyani['wastage%']

    (Xveg_biriyani_train, Xveg_biriyani_test, Yveg_biriyani_train, Yveg_biriyani_test) = \
        train_test_split(Xveg_biriyani, Yveg_biriyani, test_size=0.2, random_state=42)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(Xveg_biriyani_train, Yveg_biriyani_train)
    training_veg_biriyani_prediction = lin_reg_model.predict(Xveg_biriyani_train)
    train_error_score = metrics.r2_score(Yveg_biriyani_train,
            training_veg_biriyani_prediction)
    Yveg_biriyani_pred = lin_reg_model.predict(Xveg_biriyani_test)
    test_error_score = metrics.r2_score(Yveg_biriyani_test, Yveg_biriyani_pred)
    print ("<br/><b>Predicted Veg biryani wastage :", Yveg_biriyani_pred)


def veg_meals_wastage_pred():
    Xveg_meals = veg_meals.drop(['date', 'menu-id', 'flavor_profile',
                      'calorielevel-per-100gm', 'wastage%'], axis=1)
    Yveg_meals = veg_meals['wastage%']

    (Xveg_meals_train, Xveg_meals_test, Yveg_meals_train, Yveg_meals_test) = \
        train_test_split(Xveg_meals, Yveg_meals, test_size=0.2, random_state=42)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(Xveg_meals_train, Yveg_meals_train)
    training_veg_meals_prediction = lin_reg_model.predict(Xveg_meals_train)
    train_error_score = metrics.r2_score(Yveg_meals_train,
            training_veg_meals_prediction)
    Yveg_meals_pred = lin_reg_model.predict(Xveg_meals_test)
    test_error_score = metrics.r2_score(Yveg_meals_test, Yveg_meals_pred)
    print ("<br/><b>Predicted Veg meals wastage :", Yveg_meals_pred)
    


def desert_wastage_pred():
    Xdesert = desert.drop(['date', 'menu-id', 'flavor_profile',
                      'calorielevel-per-100gm', 'wastage%'], axis=1)
    Ydesert = desert['wastage%']

    (Xdesert_train, Xdesert_test, Ydesert_train, Ydesert_test) = \
        train_test_split(Xdesert, Ydesert, test_size=0.2, random_state=42)
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(Xdesert_train, Ydesert_train)
    training_desert_prediction = lin_reg_model.predict(Xdesert_train)
    train_error_score = metrics.r2_score(Ydesert_train,
            training_desert_prediction)
    Ydesert_pred = lin_reg_model.predict(Xdesert_test)
    test_error_score = metrics.r2_score(Ydesert_test, Ydesert_pred)
    print ("<br/><b>Predicted Desert wastage :", Ydesert_pred)

    
    
    

idli_wastage_pred()
samosa_wastage_pred()
roti_wastage_pred()
boiledegg_wastage_pred()
Chicken_curry_wastage_pred()
veg_biriyani_wastage_pred()
veg_meals_wastage_pred()
desert_wastage_pred()

