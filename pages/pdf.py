import base64
import io
import dash
from dash import Dash, dash_table, dcc, html, exceptions, ctx
from dash.dependencies import Input, Output, State
import pandas as pd

import camelot as cam
import PyPDF2
from PyPDF2 import PdfReader

from currency_converter import CurrencyConverter
import re

import spacy

nlp = spacy.load("en_core_web_sm")

dash.register_page(__name__, path='/')


cr = CurrencyConverter()

input_currency = "JPY"

number_scale = 1

number_scale = 1000

def currency_conversion(amount, input_currency, output_currency='USD'):
    converted_amount = cr.convert(amount, input_currency, output_currency)
    return converted_amount

def is_number(string):
    try:
        int(string)
        return True
    except ValueError:
        pass
    
    try:
        float(string)
        return True
    except ValueError:
        pass
    
    return False


# Callback to parse contents of a pdf
@dash.callback([Output('pdf-viewer', 'columns'),
                Output('pdf-viewer', 'data'),
                Output('pdf', 'src')],
                [Input('pdf-upload', 'contents'),
                State('pdf-upload', 'filename'),
                Input('adding-columns-button', 'n_clicks'),
                State('adding-columns-name', 'value'),
                State('pdf-viewer', 'columns'),
                Input('adding-rows-button', 'n_clicks'),
                Input('convert-currency-button', 'n_clicks'),
                Input('pdf-viewer', 'selected_cells'),
                State('pdf-viewer', 'data')],
                prevent_initial_call = True
                )
def update_table(contents, filename, n_clicks, value, existing_columns, n_clicks_row, n_clicks_convert, selected_cells, table_data):
    triggered_id = ctx.triggered_id
    # print(triggered_id)
    if triggered_id == 'pdf-upload':
        return pdf_output(contents, filename)
    elif triggered_id == 'adding-columns-button':
        return update_columns(n_clicks, value, existing_columns, table_data, contents)
    elif triggered_id == 'adding-rows-button':
        return add_row(n_clicks_row, table_data, existing_columns, contents)
    elif triggered_id == 'convert-currency-button':
        return convert_currency(n_clicks_convert, table_data, selected_cells, existing_columns, contents)
    else:
        raise exceptions.PreventUpdate

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

        return [{'name': 'Column {}'.format(i), 'id': str(i), 'deletable': True, 'renamable': True} for i in pd.concat(list_of_dfs).columns], pd.concat(list_of_dfs).to_dict('records'), contents

def update_columns(n_clicks, value, existing_columns, table_data, contents):
    if n_clicks > 0:
        existing_columns.append({
            'name': value, 'id': value, 
            'renamable': True, 'deletable': True
        })
    else:
        raise exceptions.PreventUpdate
    return existing_columns, table_data, contents

def add_row(n_clicks_row, table_data, existing_columns, contents):
    if n_clicks_row > 0:
        table_data.append({c['id']: '' for c in existing_columns})
    return existing_columns, table_data, contents

def convert_currency(n_clicks_convert, table_data, selected_cells, existing_columns, contents):
    print(table_data)
    if n_clicks_convert > 0:
        for cell in selected_cells:
            cell_value = table_data[cell['row']][cell['column_id']]
            print(cell_value)
            print(type(cell_value))
            if isinstance(cell_value, str):
                # Convert the cell value to USD
                if cell_value != "" or cell_value != None:
                    cell_value = re.sub('/[^A-Za-z0-9.\-]/','', cell_value )
                    cell_value = cell_value.replace(",",  "")
                    if is_number(cell_value):
                        cell_value = float(cell_value)
                        cell_value = cell_value*number_scale
                        converted_amount = currency_conversion(cell_value, input_currency)
                        table_data[cell['row']][cell['column_id']] = str(converted_amount)
            else:
                # Print the cell value if it's not a number
                print(cell_value) 
    return existing_columns, table_data, contents

@dash.callback([Output('new-table', 'columns'),
                Output('new-table', 'data')],
                [Input('pdf-viewer', 'selected_rows'),
                Input('get-new-table-button', 'n_clicks'),
                Input('pdf-viewer', 'data')],
                prevent_initial_call = True)
def new_table(selected_rows, n_clicks, table_data):
    if selected_rows is None:
        selected_rows = []

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'get-new-table-button' in changed_id:
        selected_rows = sorted(selected_rows)
        list_of_dicts = []
        for i in selected_rows:
            list_of_dicts.append(table_data[i])
    else:
        raise exceptions.PreventUpdate
    
    return [{'name': 'Column {}'.format(i), 'id': str(i), 'deletable': True, 'renamable': True} for i in pd.DataFrame(list_of_dicts).columns], list_of_dicts
    
@dash.callback(Output('pdf-viewer', 'selected_rows'),
                [Input('spacy-button', 'n_clicks'),
                Input('pdf-viewer', 'data')],
                prevent_initial_call = True)
def highlight_rows(n_clicks_spacy_button, table_data):
    print(table_data)
    financial_terms = ["revenue", "profit", "income", "expense", "loss", "cost", "gross profit", "gross loss", "net profit", "net loss", "EBIDTA", "total equities", "equities", "total liabilities", "liabilities", "total assets", "assets", "total current assets", "current assets", "total non-current assets", "non-current assets", "debt", "cash", "net cash flow", "cash flow"]
    row_ids = []
    if n_clicks_spacy_button > 0:
        for i in range(len(table_data)):
            text = "\n".join(list(table_data[i].values()))
            doc = nlp(text)

            for token in doc:
                if token.text.lower() in financial_terms:
                    row_ids.append(i)

        row_ids = sorted(list(dict.fromkeys(row_ids)))
    else:
        raise exceptions.PreventUpdate

    return row_ids

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
                                id='pdf-viewer',
                                row_selectable='multi', 
                                selected_columns=[], 
                                selected_rows=[]                           
                                )

pdf = html.Iframe(id='pdf', height="1067px", width="100%")

#Adding columns:
add_columns = dcc.Input(
            id='adding-columns-name',
            placeholder='Enter a column name...',
            value='',
            style={'padding': 10}
                )

extracted_table = dash_table.DataTable(editable=True,
                                        row_deletable=True,
                                        export_format='xlsx',
                                        # page_action='none',
                                        # fixed_rows={'headers': True},
                                        # style_table={'height': 500, 'overflowY': 'auto'},
                                        # style_header={'overflowY': 'auto'}
                                        id='new-table',
                                        row_selectable='multi', 
                                        selected_columns=[], 
                                        selected_rows=[]                           
                                        )

#Place into the app
layout = html.Div([html.H4('Convert PDF using Camelot and dash'),
                        pdf_load,
                        html.Br(),
                        html.H5('Preview of PDF'),
                        pdf,
                        html.H5('PDF output table'),
                        html.Div([add_columns, 
                                html.Button('Add Column', id='adding-columns-button', n_clicks=0)
                                ], style={'height': 50}),
                        pdf_table,
                        html.Button('Add Row', id='adding-rows-button', n_clicks=0),
                        html.Button('Convert Currency', id='convert-currency-button', n_clicks=0),
                        html.Button('Extract Table', id='get-new-table-button', n_clicks=0),
                        html.Br(),
                        html.H5('Extracted Table'),
                        extracted_table,
                        html.Button('Select Only Financial Data!', id='spacy-button', n_clicks=0)
                        ])
