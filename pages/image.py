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

import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


dash.register_page(__name__)


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
                State('img-viewer', 'data')],
                prevent_initial_call = True
                )
def update_table(contents, filename, n_clicks, value, existing_columns, n_clicks_row, table_data):
    triggered_id = ctx.triggered_id
    # print(triggered_id)
    if triggered_id == 'img-upload':
        return img_output(contents, filename)
    elif triggered_id == 'adding-columns-button':
        return update_columns(n_clicks, value, existing_columns, table_data, contents)
    elif triggered_id == 'adding-rows-button':
        return add_row(n_clicks_row, table_data, existing_columns, contents)

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
            words_length = len(words)
            numbers_length = len(numbers)
            for i in range(0, max(lengths_of_each_rows) - words_length - numbers_length):
                whole.append("")
            whole.extend(numbers)
            data.append(whole)

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
                                id='img-viewer'                          
                                )

image = html.Img(id='image')

# Adding columns:
add_columns = dcc.Input(
            id='adding-columns-name',
            placeholder='Enter a column name...',
            value='',
            style={'padding': 10}
                )

#Place into the app
layout = html.Div([html.H4('Convert Image using OpenCV, PyTesseract, Dash'),
                        img_load,
                        html.Br(),
                        image,
                        html.Div([add_columns, 
                                html.Button('Add Column', id='adding-columns-button', n_clicks=0)
                                ], style={'height': 50}),
                        img_table,
                        html.Button('Add Row', id='adding-rows-button', n_clicks=0)
                        ])
