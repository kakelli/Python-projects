import altair as alt
import pandas as pd

data = pd.DataFrame({
    'x':[1,2,3,4,5],
    'y':[6,7,8,9,10]
})
#Simple Line chart
'''chart = alt.Chart(data).mark_line().encode(
    x='x',
    y='y'
)

chart.save('alt1.html')
print("Chart is saved alt1.html")'''

#Simple bar chart
'''chart = alt.Chart(data).mark_bar().encode(
    x = 'x:O',
    y = 'y:Q'
)

chart.save('alt1bar.html')'''

#Tooltips and Color
'''chart = alt.Chart(data).mark_circle(size=100).encode(
    x='x',
    y='y',
    color='y',
    tooltip = ['x', 'y']
)

chart.save('newthing.html')'''

#Scatter plots with size
'''data['size'] = [30,60,90,120,150]
chart = alt.Chart(data).mark_circle().encode(
    x='x',
    y='y',
    size='size',
    color = 'y'
)

chart.save('alt2.html')'''

#Filtering Data
'''chart = alt.Chart(data['y']>15).mark_bar().encode(
    x='x:O',
    y='y:Q'
)

chart.save('alt3.html')'''

#Chart with titles
'''chart = alt.Chart(data, title='My First Chart').mark_line().encode(
    x='x',
    y='y'
)

chart.save('alt3.html')'''

#Saving Charts
'''chart = alt.Chart(data).mark_line().encode(x='x', y='y')
chart.save('sample.html')'''

#Layering charts
'''line = alt.Chart(data).mark_line(color='blue').encode(x='x', y='y')
points = alt.Chart(data).mark_point(color='red').encode(x='x', y='y')
c = line+points

c.save('alt4.html')'''

#Faceting(Subplots)
'''data = pd.DataFrame({
    'x': list(range(1,6))*2,
    'y': [10,20,30,40,50,15,25,35,45,55],
    'category': ['A']*5 +['B']*5
})

chart = alt.Chart(data).mark_bar().encode(x='x:O', y='y:Q').facet(column='category:N')
chart.save('alt5.html')'''

#Interactively (Selection):
'''brush = alt.selection_interval()

chart = alt.Chart(data).mark_point().encode(
    x='x',
    y='y',
    color=alt.condition(brush,'y:Q', alt.value('light gray'))
).add_params(brush)

chart.save('alt6.html')'''

#Linked Charts
'''selection = alt.selection_single(fields = ['x'], bind='legend')

points = alt.Chart(data).mark_point().encode(
    x='x',
    y='y',
    color = alt.condition(selection, 'x:N', alt.value('gray')).add_params(selection)
)

bars = alt.Chart(data).mark_bar().encode(
    x='x',
    y='y',
    color=alt.condition(selection, 'x:N', alt.value('gray'))
)

chart = points & bars
chart.save('alt7.html')'''

