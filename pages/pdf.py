import base64
import io
import dash
from dash import Dash, dash_table, dcc, html, exceptions, ctx
from dash.dependencies import Input, Output, State
import pandas as pd

import camelot as cam
import PyPDF2
from PyPDF2 import PdfReader


dash.register_page(__name__, path='/')


# Callback to parse contents of a pdf
@dash.callback([Output('pdf-viewer', 'columns'),
                Output('pdf-viewer', 'data')],
                [Input('pdf-upload', 'contents'),
                State('pdf-upload', 'filename'),
                Input('adding-columns-button', 'n_clicks'),
                State('adding-columns-name', 'value'),
                State('pdf-viewer', 'columns'),
                Input('adding-rows-button', 'n_clicks'),
                State('pdf-viewer', 'data')],
                prevent_initial_call = True
                )
def update_table(contents, filename, n_clicks, value, existing_columns, n_clicks_row, table_data):
    triggered_id = ctx.triggered_id
    # print(triggered_id)
    if triggered_id == 'pdf-upload':
        return pdf_output(contents, filename)
    elif triggered_id == 'adding-columns-button':
        return update_columns(n_clicks, value, existing_columns, table_data)
    elif triggered_id == 'adding-rows-button':
        return add_row(n_clicks_row, table_data, existing_columns)

def pdf_output(contents, filename):
    if contents is None:
        raise exceptions.PreventUpdate
    if contents is not None:
        file_dict = {}

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        input_pdf = io.BytesIO(decoded)

        with open("input.pdf", "wb") as f:
            base64_pdf = base64.b64encode(input_pdf.read()).decode('utf-8')
            f.write(base64.b64decode(base64_pdf))
        f.close()

        readpdf = PdfReader("input.pdf")
        totalpages = len(readpdf.pages)

        for num_page in range(1, int(totalpages)+1):
            file_dict["page_" + str(num_page)] = cam.read_pdf("input.pdf", flavor='stream', pages=str(num_page), edge_tol=500)

        list_of_dfs = []
        for k,v in file_dict.items():
            for i in range(len(file_dict[k])):
                list_of_dfs.append(file_dict[k][i].df)

        return [{'name': 'Column {}'.format(i), 'id': str(i), 'deletable': True, 'renamable': True} for i in pd.concat(list_of_dfs).columns], pd.concat(list_of_dfs).to_dict('records')

def update_columns(n_clicks, value, existing_columns, table_data):
    if n_clicks > 0:
        existing_columns.append({
            'name': value, 'id': value, 
            'renamable': True, 'deletable': True
        })
    else:
        raise exceptions.PreventUpdate
    return existing_columns, table_data

def add_row(n_clicks_row, table_data, existing_columns):
    if n_clicks_row > 0:
        table_data.append({c['id']: '' for c in existing_columns})
    return existing_columns, table_data


#Upload component:
pdf_load = dcc.Upload(id='pdf-upload',
                        children=html.Div(['Drag and Drop or ', html.A('Select PDF files')]),
                        style={'width': '90%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                        )

#Table to view output from pdf:
pdf_table = dash_table.DataTable(editable=True,
                                row_deletable=True,
                                export_format='xlsx',
                                # page_action='none',
                                # fixed_rows={'headers': True},
                                # style_table={'height': 500, 'overflowY': 'auto'},
                                # style_header={'overflowY': 'auto'}
                                id='pdf-viewer'                          
                                )

#Adding columns:
add_columns = dcc.Input(
            id='adding-columns-name',
            placeholder='Enter a column name...',
            value='',
            style={'padding': 10}
                )

#Place into the app
layout = html.Div([html.H4('Convert PDF using Camelot and dash'),
                        pdf_load,
                        html.Br(),
                        html.Div([add_columns, 
                                html.Button('Add Column', id='adding-columns-button', n_clicks=0)
                                ], style={'height': 50}),
                        pdf_table,
                        html.Button('Add Row', id='adding-rows-button', n_clicks=0)
                        ])
