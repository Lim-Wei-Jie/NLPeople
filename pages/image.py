import base64
import io
import dash
from dash import Dash, dash_table, dcc, html, exceptions, ctx
from dash.dependencies import Input, Output, State
import pandas as pd

import cv2
import pytesseract
# from img2table.document import Image
# from img2table.ocr import TesseractOCR

from currency_converter import CurrencyConverter
from forex_python.converter import CurrencyRates
import re
from datetime import datetime

import spacy

nlp = spacy.load("en_core_web_sm")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


dash.register_page(__name__)

cr = CurrencyRates()

# input_currency = "JPY"

# number_scale = 1

# number_scale = 1000

# def currency_conversion(amount, input_currency, output_currency='USD'):
#     converted_amount = cr.convert(amount, input_currency, output_currency)
#     return converted_amount

def is_number(string):
    string = string.replace(',', '')
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
@dash.callback([Output('img-viewer', 'columns'),
                Output('img-viewer', 'data'),
                Output('image', 'src')],
                [Input('img-upload', 'contents'),
                State('img-upload', 'filename'),
                Input('adding-columns-button', 'n_clicks'),
                State('adding-columns-name', 'value'),
                State('img-viewer', 'columns'),
                Input('adding-rows-button', 'n_clicks'),
                Input('convert-currency-button', 'n_clicks'),
                Input('currency-input-dropdown', 'value'),
                Input('currency-output-dropdown', 'value'),
                Input('scale-input-dropdown', 'value'),
                Input('scale-output-dropdown', 'value'),
                Input('img-viewer', 'selected_cells'),
                State('img-viewer', 'data')],
                prevent_initial_call = True
                )
def update_table(contents, filename, n_clicks, value, existing_columns, n_clicks_row, n_clicks_convert, currency_input, currency_output, scale_input, scale_output, selected_cells, table_data):
    triggered_id = ctx.triggered_id
    # print(triggered_id)
    if triggered_id == 'img-upload':
        return img_output(contents, filename)
    elif triggered_id == 'adding-columns-button':
        return update_columns(n_clicks, value, existing_columns, table_data, contents)
    elif triggered_id == 'adding-rows-button':
        return add_row(n_clicks_row, table_data, existing_columns, contents)
    elif triggered_id == 'convert-currency-button':
        return convert_currency(n_clicks_convert, table_data, selected_cells, currency_input, currency_output, scale_input, scale_output, existing_columns, contents)
    else:
        raise exceptions.PreventUpdate

def img_output(contents, filename):
    if contents is None:
        raise exceptions.PreventUpdate
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        input_img = io.BytesIO(decoded)

        with open("input.img", "wb") as f:
            base64_img = base64.b64encode(input_img.read()).decode('utf-8')
            f.write(base64.b64decode(base64_img))
        f.close()

        img = cv2.imread("input.img")

        # Perform OCR on the image
        text = pytesseract.image_to_string(img, config="--psm 6")
        # --psm 1/3/4/6 || 11/12

        # Split the extracted text into rows
        rows = text.split("\n")

        # Create a list of lists to store the data
        lengths_of_each_rows = []
        for row in rows:
            lengths_of_each_rows.append(len(row.split()))

        data = []
        for row in rows:
            words = []
            numbers = []

            if len(row) != 0:
                if re.search("[0-9]", row.split()[0]) and re.search("[0-9]", row.split()[1]):
                    numbers.append(row.split()[0])
                else:
                    words.append(row.split()[0])

            for i in range(1, len(row.split())-1):
                    if re.search("[0-9]", row.split()[i]) and re.search("[0-9]", row.split()[i+1]):
                        numbers.append(row.split()[i])
                    else:
                        words.append(row.split()[i])

            if len(row) != 0:
                numbers.append(row.split()[-1])

            whole = []
            whole.extend(words)

            temp = " ".join(whole)
            whole = [temp]

            numbers_length = len(numbers)
            for i in range(0, max(lengths_of_each_rows) - numbers_length):
                whole.append("")
            print("whole3:", whole)
            whole.extend(numbers)
            print("whole4:", whole)
            data.append(whole)
            print("data:", data)

        df = pd.DataFrame(data)

        return [{'name': 'Column {}'.format(i), 'id': str(i), 'deletable': True, 'renamable': True} for i in df.columns], df.to_dict('records'), contents


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

def convert_currency(n_clicks_convert, table_data, selected_cells, currency_input,currency_output, scale_input, scale_output, existing_columns, contents):
    print(table_data)
    if n_clicks_convert > 0:
        for cell in selected_cells:
            cell_value = table_data[cell['row']][cell['column_id']]
            print(cell_value)
            print(type(cell_value))
            if isinstance(cell_value, str):
                # Convert the cell value to USD
                if cell_value != "" or cell_value != None:
                    cell_value = re.sub('/[^A-Za-z0-9.\-]/','', cell_value)
                    cell_value = cell_value.replace(',', '')
                    print(cell_value)
                    if is_number(cell_value):
                        cell_value = float(cell_value)
                        if scale_output == "NA" or scale_input == "NA":
                            cell_value = cell_value
                        else:
                            if scale_input == "thousands" or scale_input == "millions" or scale_input == "billions":
                                # eg. 1000 > 1k, 1000000 > 1000k, 1000000000 > 1000000k
                                if scale_output == "k":
                                        cell_value = cell_value/1000
                                # eg. 1000 > 0.001m , 1000000 > 1m, 1000000000 > 1000m
                                elif scale_output == "m":
                                    cell_value = cell_value/1000000
                                # eg. 1000 > 0.000001b , 1000000 > 0.001b, 1000000000 > 1b
                                elif scale_output == "b":
                                    cell_value = cell_value/1000000000
                            
                            if scale_input == "k":
                                # eg. 1k > 1000 (real value)
                                if scale_output == "thousands":
                                    cell_value = cell_value*1000
                                elif scale_output == "k":
                                    cell_value = cell_value
                                # eg. 1k > 0.001m
                                elif scale_output == "millions" or scale_output == "m":
                                    cell_value = cell_value/1000
                                # eg. 1k > 0.000001b 
                                elif scale_output == "billions" or scale_output == "b":
                                    cell_value = cell_value/1000000
                            elif scale_input == 'm':
                                # eg. 1m > 1000k
                                if scale_output == "thousands" or scale_output == "k":
                                    cell_value = cell_value*1000
                                # eg. 1m > 1000000 (real value)
                                elif scale_output == "millions":
                                    cell_value = cell_value*1000000
                                elif scale_output == "m":
                                    cell_value = cell_value
                                # eg. 1m > 0.001b
                                elif scale_output == "billions" or scale_output == "b":
                                    cell_value = cell_value/1000
                            elif scale_input == 'b':
                                # eg. 1b > 1000000k
                                if scale_output == "thousands" or scale_output == "k":
                                    cell_value = cell_value*1000000
                                # eg. 1b > 1000m 
                                elif scale_output == "millions" or scale_output == "m":
                                    cell_value = cell_value*1000
                                # eg. 1m > 0.001b
                                elif scale_output == "b":
                                    cell_value = cell_value
                                # eg. 1b > 1000000000 (real value)
                                elif scale_output == "billions":
                                    cell_value = cell_value*1000000000
                            
                            if scale_input == "thousands":
                                if scale_output == "thousands":
                                    cell_value = cell_value
                                elif scale_output == "millions":
                                    cell_value = cell_value*0.001
                                elif scale_output == "billions":
                                    cell_value = cell_value*0.000001
                            elif scale_input == "millions":
                                if scale_output == "thousands":
                                    cell_value = cell_value * 1000
                                elif scale_output == "millions":
                                    cell_value = cell_value
                                elif scale_output == "billions":
                                    cell_value = cell_value*0.001
                            elif scale_input == "billions":
                                if scale_output == "thousands":
                                    cell_value = cell_value * 1000000
                                elif scale_output == "millions":
                                    cell_value = cell_value * 1000
                                elif scale_output == "billions":
                                    cell_value = cell_value
                        # cell_value = cell_value*scale_input
                        converted_amount = cr.convert(currency_input, currency_output, cell_value)
                        table_data[cell['row']][cell['column_id']] = str(converted_amount)
            else:
                # Print the cell value if it's not a number
                print(cell_value) 
        #print date in cell
        break_loop = False
        for i in range(len(table_data),-1, -1):
            for x in range(len(table_data[i-1]),-1,-1):
                if table_data[i-1][str(x-1)] == "":
                    table_data[i-1][str(x-1)] = "Exchage Rate from "  + str(datetime.today())
                    break_loop = True
                    break
            if break_loop:
                break
    return existing_columns, table_data, contents


@dash.callback([Output('new-table-img', 'columns'),
                Output('new-table-img', 'data')],
                [Input('img-viewer', 'selected_rows'),
                Input('get-new-table-button', 'n_clicks'),
                Input('img-viewer', 'data'),
                Input('adding-columns-button-new-table', 'n_clicks'),
                State('adding-columns-name-new-table', 'value'),
                State('new-table-img', 'columns'),
                State('new-table-img', 'data'),
                Input('adding-rows-button-new-table', 'n_clicks')],
                prevent_initial_call = True)
def update_extracted_table(selected_rows, n_clicks_get_table, table_data, n_clicks_add_column, new_column_name, existing_columns, new_table_data, n_clicks_add_row):
    triggered_id = ctx.triggered_id
    if triggered_id == 'get-new-table-button':
        return new_table(selected_rows, n_clicks_get_table, table_data)
    elif triggered_id == 'adding-columns-button-new-table':
        return update_columns_new_table(n_clicks_add_column, new_column_name, existing_columns, new_table_data)
    elif triggered_id == 'adding-rows-button-new-table':
        return add_row_new_table(n_clicks_add_row, new_table_data, existing_columns)
    else:
        raise exceptions.PreventUpdate

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

def update_columns_new_table(n_clicks_add_column, new_column_name, existing_columns, new_table_data):
    if n_clicks_add_column > 0:
        existing_columns.append({
            'name': new_column_name, 'id': new_column_name, 
            'renamable': True, 'deletable': True
        })
    else:
        raise exceptions.PreventUpdate
    return existing_columns, new_table_data

def add_row_new_table(n_clicks_add_row, new_table_data, existing_columns):
    if n_clicks_add_row > 0:
        new_table_data.append({c['id']: '' for c in existing_columns})
    return existing_columns, new_table_data

@dash.callback(Output('img-viewer', 'selected_rows'),
                [Input('spacy-button', 'n_clicks'),
                Input('img-viewer', 'data')],
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

@dash.callback(
    Output(component_id='currency-is-in-img', component_property='children'),
    [Input(component_id='scale-input-dropdown', component_property='value'),
    Input(component_id='scale-output-dropdown', component_property='value')]
)
def update_output_div(input_scale, output_scale):
    if input_scale == "NA" or output_scale == "NA":
        text = ""
    else:
        text = "The output table values are converted from " + input_scale + " to " + output_scale
    return text

# Upload component:
img_load = dcc.Upload(id='img-upload',
                        children=html.Div(['Drag and Drop or ', html.A('Select Image files')]),
                        style={'width': '90%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                        )

#Table to view output from pdf:
img_table = dash_table.DataTable(editable=True,
                                row_deletable=True,
                                export_format='xlsx',
                                id='img-viewer',
                                row_selectable='multi', 
                                selected_columns=[], 
                                selected_rows=[]                           
                                )

image = html.Img(id='image')

# Adding columns:
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
                                        id='new-table-img',
                                        row_selectable='multi', 
                                        selected_columns=[], 
                                        selected_rows=[]                           
                                        )

#Place into the app
layout = html.Div([html.H4('Convert Image using OpenCV, PyTesseract, Dash'),
                        img_load,
                        html.Br(),
                        html.H5('Preview of Image'),
                        image,
                        html.H5('Image output table'),
                        html.Div(id='currency-is-in-img'),
                        html.Div([add_columns, 
                                html.Button('Add Column', id='adding-columns-button', n_clicks=0)
                                ], style={'height': 50}),
                        img_table,
                        html.Button('Add Row', id='adding-rows-button', n_clicks=0),
                        html.Br(), html.Br(),
                        html.H5('Convert Currency'),
                        html.P('Convert from:'),
                        dcc.Dropdown(options=['USD', 'JPY', 'BGN', 'CZK', 'DKK', 'GBP', 'HUF', 'PLN', 'RON', 'SEK', 'CHF', 'ISK', 'NOK', 'TRY', 'AUD', 'BRL', 'CAD', 'CNY', 'HKD', 'IDR', 'ILS', 'INR', 'KRW', 'MXN', 'MYR', 'NZD', 'PHP', 'SGD', 'THB', 'ZAR'], value='USD', id='currency-input-dropdown'),
                        html.P('Convert to:'),
                        dcc.Dropdown(options=['USD', 'JPY', 'BGN', 'CZK', 'DKK', 'GBP', 'HUF', 'PLN', 'RON', 'SEK', 'CHF', 'ISK', 'NOK', 'TRY', 'AUD', 'BRL', 'CAD', 'CNY', 'HKD', 'IDR', 'ILS', 'INR', 'KRW', 'MXN', 'MYR', 'NZD', 'PHP', 'SGD', 'THB', 'ZAR'], value='USD', id='currency-output-dropdown'),
                        html.P('Scale from:'),
                        dcc.Dropdown(options=['NA', 'thousands', 'millions', 'billions', 'k', 'm', 'b'], id='scale-input-dropdown', value='NA'),
                        html.P('Scale to:'),
                        dcc.Dropdown(options=['NA', 'thousands', 'millions', 'billions', 'k', 'm', 'b'], id='scale-output-dropdown', value='NA'),
                        html.Button('Convert Currency', id='convert-currency-button', n_clicks=0),
                        html.Br(), html.Br(),
                        
                        html.H5('Extracted Table'),
                        html.Button('Select Only Financial Data!', id='spacy-button', n_clicks=0),
                        html.Button('Extract Table', id='get-new-table-button', n_clicks=0),
                        html.Br(), html.Br(),
                        html.Div([dcc.Input(
                                            id='adding-columns-name-new-table',
                                            placeholder='Enter a column name...',
                                            value='',
                                            style={'padding': 10}
                                                ), 
                                html.Button('Add Column', id='adding-columns-button-new-table', n_clicks=0)
                                ], style={'height': 50}),
                        extracted_table,
                        html.Button('Add Row', id='adding-rows-button-new-table', n_clicks=0)
                        ])
