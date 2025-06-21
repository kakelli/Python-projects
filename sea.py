import seaborn as sns
import matplotlib.pyplot as plt

#Loading sample dataset
'''f = sns.load_dataset('tips')
print(f.head())'''

#Creating a basic plot
'''f= sns.load_dataset('tips')
sns.scatterplot(x = 'total_bill', y = 'tip', data = f)
plt.show()'''

#Creating a line plot
'''f = sns.load_dataset('tips')
sns.lineplot(x = 'size', y = 'tip', data='f')
plt.show()'''

#Creating a bar plot
'''f = sns.load_dataset('tips')
sns.barplot(x = 'day', y='total_bill', data=f)
plt.show()'''

#Creating a histogram
'''f =sns.load_dataset('tips')
sns.histplot(f['total_bill'], bins = 50)
plt.show()'''

#Creating a box plot
'''f = sns.load_dataset('tips')
sns.boxplot(x ='day', y='total_bill', data = f)
plt.show()'''

#Color category (Hue)
'''f = sns.load_dataset('tips')
sns.scatterplot( x='total_bill', y='tip', hue='sex', data=f)
plt.show()'''

#Count plot
'''f = sns.load_dataset('tips')
sns.countplot(x='day', data=f)
plt.show()'''

#Violin plot
'''f = sns.load_dataset('tips')
sns.violinplot(x='day', y='total_bill', data=f)
plt.show()'''

#Pair plot(Multiple Scatter plots)
'''f = sns.load_dataset('tips')
sns.pairplot(f)
plt.show()'''

#Heatmap (Correlation matrix)
'''f = sns.load_dataset('tips')
corr = f.corr()
sns.heatmap(corr, annot=True,cmap='coolwarm')
plt.show()'''

#FacetGrid(Multiple plots in a grid)
'''f = sns.load_dataset('tips')
g = sns.FacetGrid(f, col='sex')
g.map(sns.histplot, 'total_bill')
plt.show()'''

#Joining plot(2D+histogram)
'''f = sns.load_dataset('tips')
sns.jointplot(x='total_bill', y='tip', data=f, kind='hex')
plt.show()'''

#Customizing Style and themes
'''f = sns.load_dataset('tips')
sns.set_style('whitegrid')  # Set style
sns.boxplot(x='day', y='tip', data=f)
plt.title('Tips per day')
plt.show()'''

#Regression plot
'''f = sns.load_dataset('tips')
sns.lmplot(x='total_bill', y='tip', data=f)
plt.show()'''

#Saving plots as image
f = sns.load_dataset('tips')
sns.histplot(f['tip'])
plt.savefig('tips_distribution.png')  # Save the plot as a PNG file
