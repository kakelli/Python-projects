import pandas as pd
import matplotlib.pyplot as plt
#Creating pandas series
'''u = [1,2,3]
print(pd.Series(u))'''

#Labels in pandas
'''y = [1,3,5,7]
print(y[1])'''

#Creating index in pandas
'''g = [1,3,5,7,9]
k = pd.Series(g, index=['a', 'b', 'c','d','e'])
print(k['c'])'''

#Dictionary in series
'''d = {
    'day1': 420,
    'day2':330,
    'day3':240
}
b = pd.Series(d, index = ['day1', 'day2'])
print(b)'''

#Creating a dataframe(multi-dimensional frame)
'''f = {
    'calories':[450,360,270],
    'duration':[50,30,45]
}
d = pd.DataFrame(f)
print(d)'''

#Locating a row in Pandas
'''d = {
    'clubs': ['FC Barcelona', 'Real Madrid', 'Manchester city', 'FC Bayern Munich'],
    'players': ['Raphinha', 'Mbappe', 'De Bruyne', 'Musiala']
}
c = pd.DataFrame(d)
print(c.loc[[0,1]])'''

#Naming the index in Pandas
'''r = {
    'Countries':['Israel', 'India', 'USA', 'England'],
    'Capitals': ['Jerusalem', 'New Delhi', 'Washington DC', 'London']
}
f = pd.DataFrame(r, index=['goat', 'child', 'mature', 'fallen'])
print(f)'''

#Loading files in Pandas(csv):
'''f = pd.read_csv("/home/johnbright/Desktop/ogdata.csv")
print(f)'''

#Printing entire DataFrame(to_string()) in pandas:
'''f = pd.read_csv("/home/johnbright/Desktop/data.csv")
d = f.to_string()
print(d)'''

#Printing entrie DataFrame(read_csv()) in pandas:
'''f = pd.read_csv("/home/johnbright/Desktop/data.csv")
print(f)'''

#Checking maximum rows in pandas(pd.options.display.max_rows())
'''print(pd.options.display.max_rows)'''

#Changing the maximum rows requirement:
'''pd.options.display.max_rows = 100
f = pd.read_csv("/home/johnbright/Desktop/data.csv")
print(pd.options.display.max_rows)'''

# Loading JSON file using to_string():
'''f = pd.read_csv("/home/johnbright/Desktop/data.json")
d = f.to_string()
print(d)'''

#Dictionary as JSON using DataFrame():

'''l = {
  "Clubs": {
      "FCB": 4,
      "FCBY": 2,
      "RMA": 0
  },
  "Countries": {
    "AFC": 3,
    "BRS": 5,
    "POR": 0  
  },
  "USA":{
    "MIA": 2,
    "CIN": 3 ,
    "LAFC": 2
  }
}
f = pd.DataFrame(l)
print(f)'''

#Viewing the data in pandas using head()- to print the first 5 rows:
'''f = pd.read_csv("/home/johnbright/Desktop/data.csv")
d = f.head()
print(d)'''

#Viewing the data in pandas using tail()- to print last 5 rows:
'''d = pd.read_csv("/home/johnbright/Desktop/data.csv")
f = d.tail()
print(f)'''

#Viewing the info about file in pandas- info():
'''f = pd.read_csv("/home/johnbright/Desktop/data.csv")
print(f.info())'''

#Deleting an empty row from data set- dropna():
'''f = pd.read_csv("/home/johnbright/Desktop/data.csv")
f.dropna(inplace = True) #inplace = True (to change the original dataset)
print(f.to_string())'''

#Replacing empty rows - fillna():
'''f = pd.read_csv("/home/johnbright/Desktop/data.csv")
f.fillna(69, inplace=True)
print(f.to_string)'''

#Cleaning empty values- dropna():
'''f = {
    "Marks": [60,40,50,80,90,pd.NA, 90,20,45,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A'],
    "Avg":[80.2,63.2,90.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9],
    "DOA": ['2024/03/23', '2022/12/09', '2023/4/23', 2020/11/19, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01']
}
d = pd.DataFrame(f)
k = d.dropna()
print(k)'''

#Replacing empty values- fillna():
'''f = {
    "Marks": [60,40,50,80,90,pd.NA, 90,20,45,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A'],
    "Avg":[80.2,63.2,90.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9],
    "DOA": ['2024/03/23', '2022/12/09', '2023/4/23', 2020/11/19, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01']
}
a = pd.DataFrame(f)
s = a.fillna({'Marks':69})
print(s)'''

#Replacing using mean, median and mode:
'''f = {
    "Marks": [60,40,50,80,90,69, 90,20,45,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A'],
    "Avg":[80.2,63.2,90.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9],
    "DOA": ['2024/03/23', '2022/12/09', '2023/4/23', 2020/11/19, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01']
}
e = pd.DataFrame(f)
d = e['Marks'].mean()
j = e.fillna({'Marks':d})'''

#Changing of data of wrong date- to_datetime():
'''f = {
    "Marks": [60,40,50,80,90,pd.NA, 90,20,45,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A'],
    "Avg":[80.2,63.2,90.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9],
    "DOA": ['2024/03/23', pd.NA, '2023/4/23', 20201119, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01']
}

d = pd.DataFrame(f)
d['DOA'] = pd.to_datetime(d['DOA'], format = 'mixed')
print(d.to_string())'''

#Removal of Date- dropna():
'''f = {
    "Marks": [60,40,50,80,90,pd.NA, 90,20,45,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A'],
    "Avg":[80.2,63.2,90.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9],
    "DOA": ['2024/03/23', pd.NA, '2023/4/23', 20201119, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01']
}
d = pd.DataFrame(f)
print(d.dropna(subset=['DOA']))'''

#Replacing wrong values using -f.loc():
'''f = {
    "Marks": [60,40,50,80,90,pd.NA, 90,20,45,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A'],
    "Avg":[80.2,63.2,190.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9],
    "DOA": ['2024/03/23', '2022/12/09', '2023/4/23', 2020/11/19, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01']
}
p =  pd.DataFrame(f)
p.loc[2,"Avg"] = 90.5
print(p.to_string())'''

#Replacing by a loop- index():
'''f = {
    "Marks": [60,40,50,80,90,pd.NA, 90,20,45,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A'],
    "Avg":[80.2,63.2,90.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9],
    "DOA": ['2024/03/23', '2022/12/09', '2023/4/23', 2020/11/19, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01']
}
g = pd.DataFrame(f)
for i in g.index():
    if g.loc[i, 'Avg'] > 100.0:
        g.loc[i,"Avg"] = 100.0
    else:
        print("No elemnts greater than 100")'''

#Checking Duplicate data- duplicated():
'''f = {
    "Marks": [60,40,50,80,90,pd.NA, 90,20,45,69,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A','A'],
    "Avg":[80.2,63.2,90.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9,95.9],
    "DOA": ['2024/03/23', '2022/12/09', '2023/4/23', 2020/11/19, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01','2025/01/01']
}
h = pd.DataFrame(f)
print(h.duplicated())'''

#Removing Duplicates - drop_duplicates():
'''f = {
    "Marks": [60,40,50,80,90,pd.NA, 90,20,45,69,69],
    "Section": ["A", 'B','C','A','D','E','E','E',pd.NA,'A','A'],
    "Avg":[80.2,63.2,90.5,36.3,pd.NA,75.9,63.3,69.9,98.6,95.9,95.9],
    "DOA": ['2024/03/23', '2022/12/09', '2023/4/23', 2020/11/19, '2023/07/18', '2019/09/11', '2018/11/30','2023/08/19', '2024/03/31','2025/01/01','2025/01/01']
}
d = pd.DataFrame(f)
d.drop_duplicates(inplace=True)
print(d)'''

#Finding Correlation in dataset- corr()- does not apply to string:
'''f = {
    "Marks": [60,40,50,80,90,87, 90,20,45,69,69],
    "Avg":[80.2,63.2,90.5,36.3,45.2,75.9,63.3,69.9,98.6,95.9,95.9]
}
g = pd.DataFrame(f)
print(g.corr())'''

#Plotting data set using- plot():
'''f = {
    "Marks": [60,40,50,80,90,87, 90,20,45,69,69],
    "Avg":[80.2,63.2,90.5,36.3,45.2,75.9,63.3,69.9,98.6,95.9,95.9],
    "Internals":[12,12,9,19,20,13,14,5,6,13,16]
}
d = pd.DataFrame(f)
d.plot()
plt.show()'''

#Scattering data set using -plot():
'''f = pd.read_csv('/home/johnbright/Desktop/data.csv')
f.plot(kind='scatter', x='Duration', y='Maxpulse')
plt.show()'''

#Histogram data set using -kind():
'''f = pd.read_csv('/home/johnbright/Desktop/data.csv')
f['Duration'].plot(kind='hist')
plt.show()'''