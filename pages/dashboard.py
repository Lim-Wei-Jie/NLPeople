import base64
import datetime
import io

import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
from dash import Dash, dash_table, dcc, html, Input, Output, State, MATCH, no_update  # pip install dash (version 2.0.0 or higher)

dash.register_page(__name__)


layout = html.Div([html.H4('Upload Multiple Saved Excel files to view in Dashboard'),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select CSV or Excel Files')
                ]),
                style={
                    'width': '90%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Br(),

            html.Div(id='output-data-upload', children=[]),
            ])

# Upload CSV and Excel sheets to the app and create the tables----------------------------------------------------------
@dash.callback(Output('output-data-upload', 'children'),
                Input('upload-data', 'contents'),
                State('upload-data', 'filename'),
                State('upload-data', 'last_modified'),
                State('output-data-upload', 'children'),
                prevent_initial_call=True
)
def update_output(contents, filename, date, children):
    # part of the code snippet is from https://dash.plotly.com/dash-core-components/upload
    if contents is not None:
        for i, (c, n, d) in enumerate(zip(contents, filename, date)):

            content_type, content_string = contents[i].split(',')

            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename[i]:
                    # Assume that the user uploaded a CSV file
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in filename[i]:
                    # Assume that the user uploaded an excel file
                    df = pd.read_excel(io.BytesIO(decoded))

                # Some cleaning
                df = df.T
                new_header = df.iloc[0]
                df = df[1:]
                df.columns = new_header

                # Create the tables and empty graphs
                children.append(html.Div([
                    html.H5(filename[i]),

                    dash_table.DataTable(
                        df.to_dict('records'),
                        [{'name': i, 'id': i, 'selectable':True} for i in df.columns],
                        page_size=5,
                        filter_action='native',
                        column_selectable='single',
                        selected_columns=[df.columns[1]], # preselect the 2nd columns
                        style_table={'overflowX': 'auto'},
                        id={'type': 'dynamic-table',
                            'index': i},
                    ),
                    html.P('Type of Graph:'),
                    dcc.Dropdown(id={'type':'type_of_graph', 'index':i},
                                options=['bar', 'line'],
                                multi=False,
                                value='line',
                                style={'width': '40%'}),
                    
                    dcc.Graph(
                        id={
                            'type': 'dynamic-graph',
                            'index': i
                        },
                        figure={}
                    ),

                    # # For debugging
                    # html.Div('Raw Content'),
                    # html.Pre(contents[i][0:200] + '...', style={
                    #     'whiteSpace': 'pre-wrap',
                    #     'wordBreak': 'break-all'
                    # }),
                    html.Hr()
            ]))

            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
        return children
    else:
        return ""


# Build the graphs from the filtered data in the Datatable--------------------------------------------------------------
@dash.callback(Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure'),
                Input({'type': 'dynamic-table', 'index': MATCH}, 'derived_virtual_indices'),
                Input({'type': 'dynamic-table', 'index': MATCH}, 'selected_columns'),
                State({'type': 'dynamic-table', 'index': MATCH}, 'data'),
                Input({'type': 'type_of_graph', 'index': MATCH}, 'value')
)
def create_graphs(filtered_data, selected_col, all_data, type_of_graph):
    if filtered_data is not None:
        dff = pd.DataFrame(all_data)
        dff = dff[dff.index.isin(filtered_data)]

        if selected_col[0] == dff.columns[0]:
            return no_update
        else:
            if type_of_graph == 'line':
                fig = px.line(dff,
                            x = dff.columns[0],
                            y = selected_col[0])
            elif type_of_graph == 'bar':
                fig = px.bar(dff,
                            x = dff.columns[0],
                            y = selected_col[0])
            return fig


