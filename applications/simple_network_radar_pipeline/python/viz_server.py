import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from flask_socketio import SocketIO
import numpy as np

app = dash.Dash(__name__)
server = app.server
socketio = SocketIO(server)

data = np.array([])

app.layout = html.Div([
    html.H1("Real-time 1D Array Visualization"),
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='graph-update', interval=1000, n_intervals=0)
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-update', 'n_intervals')]
)
def update_graph(n):
    global data
    trace = go.Scatter(y=data, mode='lines+markers')
    return {'data': [trace], 'layout': go.Layout(title='1D Array Data')}

@socketio.on('update_data')
def handle_update(new_data):
    global data
    data = np.append(data, new_data)
    if len(data) > 100:
        data = data[-100:]

if __name__ == '__main__':
    socketio.run(app.server, debug=True, port=8050)
