import dash
from dash import Dash, html, dcc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        # Framework of the main app
        html.H1("Python Multipage App with Dash", style={'textAlign':'center'}),
        html.Div([
            dcc.Link(page['name'] + "  |  ", href=page['path'])
            for page in dash.page_registry.values()
        ]),
        html.Hr(),

        # Content of each page
        dash.page_container
    ]
)



if __name__ == '__main__':
    app.run_server(debug=True)