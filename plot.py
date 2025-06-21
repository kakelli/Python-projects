import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#Simple line plot- plotly.graph_objects
'''fig = go.Figure(data=go.Scatter(x=[1,2,3], y=[10,15,20]))
fig.show()'''

#Bar chart- plotly.express
'''data = {
    "Clubs":["FC Barcelona", "Real Madrid", "Manchester United"],
    "Count":[3,0,0]
}

fig = px.bar(data, x = "Clubs", y="Count")
fig.show()'''

#Pie Chart- plotly.express
'''fig = px.pie(name=["Python", "Java", "C++"], values = [50,60,70])
fig.show()'''

#Line plot with express 
'''f = {
    "Day": ["Mon", "Tue", "Wed"],
    "Sales": [100, 250,150]
}
g = pd.DataFrame(f)

fig = px.line(g, x="Day", y="Sales", title="Sales over days")
fig.show()'''

#Customize layout
'''f = {
    "Day": ["Mon", "Tue", "Wed"],
    "Sales": [100, 250, 150]
}
g=pd.DataFrame(f)
fig = px.line(g, x='Day', y='Sales')
fig.update_layout(
    title='My Chart',
    xaxis_title='X Axis',   
    yaxis_title='Y Axis',
    template = 'plotly_dark'
)
fig.show()'''

#Add markers to plot
'''fig = go.Figure()

fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6], mode='lines+markers', name = 'Line with markers'))
fig.show()'''

#Multiple Traces
'''fig = go.Figure()

fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6], name = 'A'))
fig.add_trace(go.Scatter(x=[1,2,3], y=[6,5,4], name = 'B'))

fig.show()'''

#Horizontal Bar Chart

'''fig = px.bar(x=[10,20,30], y=['A', 'B', 'C'], orientation='h')
fig.show()'''

#Subplots
'''fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Bar(x=[1,2], y=[3,4]), row=1, col=1)
fig.add_trace(go.Scatter(x=[1,2],y=[5,6]), row=1, col=2)

fig.show()'''

#Geo charts(Maps)
'''f = px.data.gapminder().query("year == 2010")

fig = px.choropleth(f, locations = 'iso_alpha', color = 'lifeExp', hover_name='country')
fig.show()'''

#3D Scatter plot
'''fig = go.Figure(data=go.Surface(z=[[1,2],[3,4]]))
fig.show()'''

#Animation
'''f = px.data.gapminder()

fig = px.scatter(f, x='gdpPercap', y='lifeExp', animation_frame='year', size = 'pop', color='continent', hover_name='country', log_x=True)
fig.show()'''

