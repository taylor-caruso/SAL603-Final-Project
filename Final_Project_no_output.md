```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
%matplotlib notebook
np.set_printoptions(threshold = np.inf)
```


```python
#Data was pulled from hockey-reference.com
nhl2009 = pd.read_csv('nhl09.csv')
nhl2010 = pd.read_csv('nhl10.csv')
nhl2011 = pd.read_csv('nhl11.csv')
nhl2012 = pd.read_csv('nhl12.csv')
nhl2013 = pd.read_csv('nhl13.csv')
nhl2014 = pd.read_csv('nhl14.csv')
nhl2015 = pd.read_csv('nhl15.csv')
nhl2016 = pd.read_csv('nhl16.csv')
nhl2017 = pd.read_csv('nhl17.csv')
nhl2018 = pd.read_csv('nhl18.csv')
```


```python
nhl2009['Season'] = 2009
nhl2010['Season'] = 2010
nhl2011['Season'] = 2011
nhl2012['Season'] = 2012
nhl2013['Season'] = 2013
nhl2014['Season'] = 2014
nhl2015['Season'] = 2015
nhl2016['Season'] = 2016
nhl2017['Season'] = 2017
nhl2018['Season'] = 2018
```


```python
nhldata = pd.concat([nhl2009,nhl2010,nhl2011,nhl2012,nhl2013,nhl2014,nhl2015,nhl2016,nhl2017,nhl2018]).reset_index(drop = True)
nhldata
```


```python
nhldata['Playoffs'] = 0
nhldata.loc[(nhldata.Team.str.contains("*", regex = False)), "Playoffs"] = 1
nhldata.Team = nhldata.Team.str.replace("*","", regex = False)
nhldata
```


```python
nhldata = nhldata.drop(['Rk','OL','SOW','GP','GF','PP','GA','PTS%','SHA','oPIM/G','SOL','SH','SRS','PPOA','SOS','PPO',
                        'PPA','SO','SA','Team','Season'], axis = 1)
nhldata
```


```python
fig = plt.figure(figsize = (8,8))
ax = fig.gca()
nhldata.hist(ax = ax)
fig.suptitle('Distributions of the Explanatory and Dependent Variables')
plt.tight_layout()
plt.show()
```


```python
plt.savefig('histograms.pdf')
```


```python
nhldata = nhldata.rename(columns={"PP%":"PP_pct","PIM/G":"PIMperG","GA/G":"GAperG","PK%":"PK_pct","GF/G":"GFperG",
                                  "S%":"S_pct", "SV%":"SV_pct"})
nhldata
```


```python
nhldata.isnull().sum()
```


```python
y = nhldata['Playoffs']
x = nhldata.drop(['Playoffs'], axis = 1)
```


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 11)
print(x_train)
print(x_test)
```


```python
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print(x_train)
```


```python
model = LogisticRegression(random_state = 11)
model.fit(x_train, y_train)
```


```python
print(model.intercept_)
```


```python
pd.DataFrame(model.coef_[0], x.columns, columns = ['Coeff'])
```


```python
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```


```python
accuracies = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
```


```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('MSE (Mean-Squared-Error): %s' %mse)
print('RMSE (Root-MSE): %s' %rmse)
print('R2 score: %s' %r2)
```


```python
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot = True, cmap = "YlGnBu" ,fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Log Model Confusion Matrix', y = 1.1)
plt.ylabel('Actual Outcome')
plt.xlabel('Predicted Outcome')
plt.tight_layout()
```


```python
plt.savefig('confusionmatrix.pdf')
```


```python
f, ax = plt.subplots(figsize = (8, 7))
corr = nhldata.corr()
sns.heatmap(corr, cmap = sns.diverging_palette(220, 10, as_cmap = True), vmin = -1.0, vmax = 1.0, square = True, 
            ax = ax)
plt.title('Correlation Heat Map')
plt.tight_layout()
```


```python
plt.savefig('corrplot.pdf')
```
