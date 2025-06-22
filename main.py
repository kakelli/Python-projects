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
chart = alt.Chart(data, title='My First Chart').mark_line().encode(
    x='x',
    y='y'
)

chart.save('alt3.html')