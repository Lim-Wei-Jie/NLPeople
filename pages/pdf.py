import base64
import io
import dash
from dash import Dash, dash_table, dcc, html, exceptions, ctx
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

import camelot as cam
import PyPDF2
from PyPDF2 import PdfReader

from forex_python.converter import CurrencyRates
import re
from datetime import datetime

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

dash.register_page(__name__, path='/')

cr = CurrencyRates()

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
                Input('currency-input-dropdown', 'value'),
                Input('currency-output-dropdown', 'value'),
                Input('convert-scale-button', 'n_clicks'),
                Input('scale-input-dropdown', 'value'),
                Input('scale-output-dropdown', 'value'),
                Input('pdf-viewer', 'selected_cells'),
                State('pdf-viewer', 'data')],
                prevent_initial_call = True
                )
def update_table(contents, filename, n_clicks, value, existing_columns, n_clicks_row, n_clicks_convert, currency_input, currency_output, n_clicks_scale, scale_input, scale_output, selected_cells, table_data):
    triggered_id = ctx.triggered_id
    # print(triggered_id)
    if triggered_id == 'pdf-upload':
        return pdf_output(contents, filename)
    elif triggered_id == 'adding-columns-button':
        return update_columns(n_clicks, value, existing_columns, table_data, contents)
    elif triggered_id == 'adding-rows-button':
        return add_row(n_clicks_row, table_data, existing_columns, contents)
    elif triggered_id == 'convert-currency-button':
        return convert_currency(n_clicks_convert, table_data, selected_cells, currency_input, currency_output, existing_columns, contents)
    elif triggered_id == 'convert-scale-button':
        return convert_scale(n_clicks_scale, table_data, selected_cells, scale_input, scale_output, existing_columns, contents)
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
    print("update_columns existing_columns:", existing_columns)
    print("existing_columns: ", existing_columns)
    existing_column_ids = []
    for i in range(len(existing_columns)):
        existing_column_ids.append(existing_columns[i]['id'])
    print("existing_column_ids: ", existing_column_ids)

    if n_clicks > 0:
        existing_columns.append({
            'name': value, 'id': str(int(existing_column_ids[-1])+1), 
            'renamable': True, 'deletable': True
        })
    else:
        raise exceptions.PreventUpdate
    return existing_columns, table_data, contents

def add_row(n_clicks_row, table_data, existing_columns, contents):
    if n_clicks_row > 0:
        table_data.append({c['id']: '' for c in existing_columns})
    return existing_columns, table_data, contents

def convert_currency(n_clicks_convert, table_data, selected_cells, currency_input, currency_output, existing_columns, contents):
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
                        converted_amount = cr.convert(currency_input, currency_output, cell_value)
                        table_data[cell['row']][cell['column_id']] = str(converted_amount)
            else:
                # Print the cell value if it's not a number
                print(cell_value) 
        
        if currency_input != currency_output:
            exchange_rate_time = str(datetime.today())
            if n_clicks_convert == 1:
                table_data.append({c['id']: '' for c in existing_columns})
                num_rows = len(table_data)
                num_cols = len(table_data[0])
                table_data[num_rows-1][num_cols-1] = "Exchange Rate from: " + exchange_rate_time
            else:
                num_rows = len(table_data)
                num_cols = len(table_data[0])
                table_data[num_rows-1][num_cols-1] = "Exchange Rate from: " + exchange_rate_time
    return existing_columns, table_data, contents

def convert_scale(n_clicks_scale, table_data, selected_cells, scale_input, scale_output, existing_columns, contents):
    if n_clicks_scale > 0:
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
                            if scale_input == "real number":
                                # eg. 1000 > 1k
                                if scale_output == "k":
                                        cell_value = cell_value/1000
                                # eg. 1000 > 0.001m 
                                elif scale_output == "m":
                                    cell_value = cell_value/1000000
                                # eg. 1000 > 0.000001b 
                                elif scale_output == "b":
                                    cell_value = cell_value/1000000000
                                elif scale_output == "real number":
                                    cell_value = cell_value
                            
                            elif scale_input == "k":
                                # eg. 1k > 1000 (real value)
                                if scale_output == "real number":
                                    cell_value = cell_value*1000
                                elif scale_output == "k":
                                    cell_value = cell_value
                                # eg. 1k > 0.001m
                                elif scale_output == "m":
                                    cell_value = cell_value/1000
                                # eg. 1k > 0.000001b 
                                elif scale_output == "b":
                                    cell_value = cell_value/1000000
                            
                            elif scale_input == 'm':
                                # eg. 1m > 1000k
                                if scale_output == "k":
                                    cell_value = cell_value*1000
                                # eg. 1m > 1000000 (real value)
                                elif scale_output == "real number":
                                    cell_value = cell_value*1000000
                                elif scale_output == "m":
                                    cell_value = cell_value
                                # eg. 1m > 0.001b
                                elif scale_output == "b":
                                    cell_value = cell_value/1000
                            
                            elif scale_input == 'b':
                                # eg. 1b > 1000000k
                                if scale_output == "k":
                                    cell_value = cell_value*1000000
                                # eg. 1b > 1000m 
                                elif scale_output == "m":
                                    cell_value = cell_value*1000
                                # eg. 1m > 0.001b
                                elif scale_output == "b":
                                    cell_value = cell_value
                                # eg. 1b > 1000000000 (real value)
                                elif scale_output == "real number":
                                    cell_value = cell_value*1000000000
                        # cell_value = cell_value*scale_input
                        table_data[cell['row']][cell['column_id']] = str(cell_value)
            else:
                # Print the cell value if it's not a number
                print(cell_value) 
    return existing_columns, table_data, contents

@dash.callback([Output('new-table', 'columns'),
                Output('new-table', 'data')],
                [Input('pdf-viewer', 'selected_rows'),
                Input('get-new-table-button', 'n_clicks'),
                Input('pdf-viewer', 'data'),
                Input('adding-columns-button-new-table', 'n_clicks'),
                State('adding-columns-name-new-table', 'value'),
                State('new-table', 'columns'),
                State('new-table', 'data'),
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

def new_table(selected_rows, n_clicks_get_table, table_data):
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
    print("existing_columns: ", existing_columns)
    existing_column_ids = []
    for i in range(len(existing_columns)):
        existing_column_ids.append(existing_columns[i]['id'])
    print("existing_column_ids: ", existing_column_ids)

    if n_clicks_add_column > 0:
        existing_columns.append({
            'name': new_column_name, 'id': str(int(existing_column_ids[-1])+1), 
            'renamable': True, 'deletable': True
        })
    else:
        raise exceptions.PreventUpdate
    return existing_columns, new_table_data

def add_row_new_table(n_clicks_add_row, new_table_data, existing_columns):
    if n_clicks_add_row > 0:
        new_table_data.append({c['id']: '' for c in existing_columns})
    return existing_columns, new_table_data


@dash.callback(Output('financial-terms-cols-boxes', 'options'),
                [State('pdf-viewer', 'data'),
                Input('spacy-button', 'n_clicks')],
                prevent_initial_call = True)
def shortlisted_financial_term_columns_for_user_selection(table_data, n_clicks):

    if n_clicks > 0:

        stemmed_master_list = ["revenue", "cost", "gross", "profit", "loss", "net", "ebitda",
                                "equity", "asset", "debt", "cash", "liability", "rev"]
        
        # go thru every col (DONE)
        # for each col, check the rows (DONE)
        # might need to apply stopword check to remove unnecessary words
        # tokenise the words in the rows of that col (DONE)
        # check if each token-word is in the masterlist (DONE)
        # for every col, keep track of the number of instances in a list (DONE)
        # if the number of instances is the greatest, take that col as the one with the financial metrics

        # determine the column with the financial terms --> get user to select the 1 column with the financial metric out of top 5
            # drop all other unnecessary columns???
            # drop all the columns before the financial list column?
        # make sure all the empty rows are filled in with the financial terms above
        # highlight the rows we need based on selected column (DONE)

        my_check = []
        instances_of_fin_terms_in_cols = []

        # get column names in list of dicts --> first key of every list
        col_names_list = []
        for colname in table_data[0]:
            print("check colname", colname)
            print("col_names_list", col_names_list)
            col_names_list.append(colname)


        for col_name in col_names_list:
            number_of_instances_in_col = 0
            for i in range(len(table_data)): # each obj in the list represents a row
                row_cell = table_data[i][str(col_name)] # this is name retrieved for every row
                print('what is row_cell: ', row_cell)
                if row_cell != "" and row_cell != None:
                    print("check row cell type: ", row_cell, type(row_cell), "row num:", i, "col_name:", col_name)
                    tokenised_row_cell = row_cell.split(" ")
                    for token in tokenised_row_cell:
                        #remove special ch
                        #convert to lower
                        print("check token:", token, "!!!col name:", col_name)
                        clean_token_v1 = re.sub(r'[^a-zA-Z]', '', token)
                        clean_token_v2 = clean_token_v1.lower()
                        clean_token_v3 = lemmatizer.lemmatize(clean_token_v2)
                        for keyword in stemmed_master_list:
                            if keyword in clean_token_v3:
                                my_check.append(token)
                                number_of_instances_in_col += 1
            instances_of_fin_terms_in_cols.append(number_of_instances_in_col)

            print("final list: ", instances_of_fin_terms_in_cols)
            print("my_check: ", my_check)   

            # for every non-zero instance in instances_of_fin_terms_in_cols, find dictionary and add into list_of_dicts.
            list_of_dicts = table_data
            checkbox_options = {}
            for i in range(len(instances_of_fin_terms_in_cols)):
                if instances_of_fin_terms_in_cols[i] > 0:
                    checkbox_options[col_names_list[i]] = col_names_list[i]

            print("list of dicts", list_of_dicts)

    else:
        raise exceptions.PreventUpdate       
                
    return checkbox_options

@dash.callback([Output('pdf-viewer', 'selected_rows')],
                [Input('financial-terms-cols-boxes', 'value'),
                Input('pdf-viewer', 'data'),
                ],
                prevent_initial_call = True)
def shortlisted_financial_term_columns_for_user_selection(value, table_data):
    # if len(value) != 0:
    if value is not None:
        print("value", value)
        print("data for shortlisted-fin-terms-cols", table_data)

    stemmed_master_list = ["revenue", "cost", "gross", "profit", "loss", "net", "ebitda",
                                "equity", "asset", "debt", "cash", "liability", "rev"]

    # get column names in list of dicts --> first key of every list
    col_names_list = []
    for colname in table_data[0]:
        col_names_list.append(colname)

    #zoom into the column name user selected
    #check every row in that column against stemmed_master_list
    #grab rowid

    grab_row_id = []
    count_for_row = 0

    if value is not None: # if user made a radio button selection (chose a financial metric column)
    # if len(value) != 0:
        for i in range(len(table_data)):
            print("da value", value)
            user_selected_value = value[0]
            row_cell = table_data[i][user_selected_value]
            print("myyy i", i)
            print("myyy value", str(value))

            # fill in the empty row cells with possible financial metrics

            if row_cell == "":
                print("went in")
                value_position = col_names_list.index(user_selected_value)
                print("value position", value_position)

                new_user_selected_value = value_position+1  #position of the right next cell (of the empty cell)

                #if column name (position) is not the last column, we safely select the right next cell
                if int(user_selected_value) < len(col_names_list)-1:
                    right_row_cell = table_data[i][str(new_user_selected_value)]
                    if right_row_cell != "":
                        print("right row cell", right_row_cell)
                        row_cell = right_row_cell

                #if this is not the first row (btw i indicates row value), we can safely select the row above
                #row above's value is i-1
                if i > 0:
                    up_row_cell = table_data[i-1][user_selected_value]                        
                    if up_row_cell != "":
                        print("up row cell", up_row_cell)
                        row_cell = up_row_cell

            if row_cell != "" and row_cell != None:
                # print("row: ", row)
                # print("row type", type(row))
                tokenised_row_cell = row_cell.split(" ")
                for token in tokenised_row_cell:
                    #remove special ch
                    #convert to lower
                    clean_token_v1 = re.sub(r'[^a-zA-Z]', '', token)
                    clean_token_v2 = clean_token_v1.lower()
                    clean_token_v3 = lemmatizer.lemmatize(clean_token_v2)
                    for keyword in stemmed_master_list:
                        if keyword == clean_token_v3:
                            # my_check.append(token)
                            # number_of_instances_in_col += 1
                            if count_for_row not in grab_row_id:
                                grab_row_id.append(count_for_row)
            count_for_row += 1
        print("grab_row_id", grab_row_id)

    else:
        raise exceptions.PreventUpdate
    return [grab_row_id]

@dash.callback(
    Output(component_id='currency-is-in', component_property='children'),
    [Input(component_id='scale-input-dropdown', component_property='value'),
    Input(component_id='scale-output-dropdown', component_property='value')],
    prevent_initial_call = True)
def update_output_div(input_scale, output_scale):
    if input_scale == "NA" or output_scale == "NA":
        text = ""
    else:
        text = "The output table values are converted from " + input_scale + " to " + output_scale
    return text

@dash.callback(
    Output(component_id='line-container', component_property='children'),
    [Input(component_id='new-table', component_property="derived_virtual_data"),
    Input(component_id='new-table', component_property='derived_virtual_selected_rows'),
    Input('new-table', 'active_cell')],
    prevent_initial_call = True)
def update_line(all_rows_data, slctd_row_indices, active_cell):
    if len(all_rows_data) != 0:
        dff = pd.DataFrame(all_rows_data)
        print("dff: ", dff)
        print("dff columns: ", dff.columns)
        print("dff columns names: ", dff.columns.values)
        print("dff[0]: ", dff.head(1))

        print("dff here: ", dff)
        transposed_dff = dff.T
        print("transposed dff: ", transposed_dff)
        new_df = transposed_dff.iloc[1:,:]
        new_df.columns = transposed_dff.iloc[0, :]
        print("new_df: ", new_df)
        print("new_df columns: ", new_df.columns)

        # colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
        #             for i in range(len(new_df))]
        # print("new_df: ", new_df)

        if active_cell is not None and len(new_df)>1 and "Quarters" in new_df:
            if active_cell["row"]!=0:
                row = active_cell["row"]
                print("row: ", row)
                # metrics = []
                # for i in range(1, len(dff)):
                #     metrics.append(dff[i][0])
                # print("metrics: ", metrics)
                # print("metrics[row] ", metrics[row])

                return [
                    dcc.Graph(id='graph',
                                figure=px.line(
                                    data_frame = new_df, 
                                    x = "Quarters",
                                    y = new_df.columns[row]
                                    # labels = {"Revenue": "% of Pop took online course"}
                                ).update_layout(autotypenumbers='convert types')
                                # ).update_layout(showlegend=False, xaxis={'categoryorder': 'total ascending'})
                                # .update_traces(marker_color=colors, hovertemplate="<b>%{y}%</b><extra></extra>")
                                )
                ]


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
                                selected_rows=[],
                                style_table={'height':'1000px', 'overflowY': 'auto'},
                                style_cell={                # ensure adequate header width when text is shorter than cell's text
                                        'minWidth': 95, 'maxWidth': 95, 'width': 95
                                    },
                                style_data={                # overflow cells' content into multiple lines
                                        'whiteSpace': 'normal',
                                        'height': 'auto'
                                    }                      
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
                                        selected_rows=[],
                                        column_selectable='multi',
                                        style_cell={                # ensure adequate header width when text is shorter than cell's text
                                        'minWidth': 95, 'maxWidth': 95, 'width': 95
                                            },
                                        style_data={                # overflow cells' content into multiple lines
                                                'whiteSpace': 'normal',
                                                'height': 'auto'
                                            }  
                                        )

#Checklist for user to select 1 correct column with financial terms
financial_terms_cols_checklist = dcc.RadioItems(
                                    id='financial-terms-cols-boxes', 
                                    inline=True
                                    )

#Place into the app
layout = html.Div([html.H4('Convert PDF using Camelot and Dash'),
                        pdf_load,
                        html.Br(),
                        html.Div([
                            html.Div([
                                html.H5('Preview of PDF'),
                                pdf,
                            ], style={'width': '42%', 'display': 'inline-block'}),
                            html.Div(style={'width': '4%', 'display': 'inline-block'}),
                            html.Div([
                                html.H5('PDF Output Table'),
                                html.Div(id='currency-is-in'),
                                html.Div([add_columns, 
                                        html.Button('Add Column', id='adding-columns-button', n_clicks=0)
                                        ], style={'height': 50}),
                                pdf_table,
                                html.Button('Add Row', id='adding-rows-button', n_clicks=0),
                                html.Button('Get Columns with Metrics!', id='spacy-button', n_clicks=0),
                                html.Button('Extract Table', id='get-new-table-button', n_clicks=0),
                                html.P("From the shortlisted columns below, please choose 1 column which contains the financial metrics."),
                                financial_terms_cols_checklist
                                ], style={'width': '42%', 'display': 'inline-block'}),
                            html.Div(style={'width': '2%', 'display': 'inline-block'}),
                            html.Div([
                                html.H5('Convert Currency'),
                                html.P('Convert from:'),
                                dcc.Dropdown(options=['USD', 'JPY', 'BGN', 'CZK', 'DKK', 'GBP', 'HUF', 'PLN', 'RON', 'SEK', 'CHF', 'ISK', 'NOK', 'TRY', 'AUD', 'BRL', 'CAD', 'CNY', 'HKD', 'IDR', 'ILS', 'INR', 'KRW', 'MXN', 'MYR', 'NZD', 'PHP', 'SGD', 'THB', 'ZAR'], value='USD', id='currency-input-dropdown'),
                                html.P('Convert to:'),
                                dcc.Dropdown(options=['USD', 'JPY', 'BGN', 'CZK', 'DKK', 'GBP', 'HUF', 'PLN', 'RON', 'SEK', 'CHF', 'ISK', 'NOK', 'TRY', 'AUD', 'BRL', 'CAD', 'CNY', 'HKD', 'IDR', 'ILS', 'INR', 'KRW', 'MXN', 'MYR', 'NZD', 'PHP', 'SGD', 'THB', 'ZAR'], value='USD', id='currency-output-dropdown'),
                                html.Button('Convert', id='convert-currency-button', n_clicks=0),
                                html.Br(), html.Br(),
                                html.H5('Convert Scale'),
                                html.P('Scale from:'),
                                dcc.Dropdown(options=['NA', 'real number', 'k', 'm', 'b'], id='scale-input-dropdown', value='NA'),
                                html.P('Scale to:'),
                                dcc.Dropdown(options=['NA', 'real number', 'k', 'm', 'b'], id='scale-output-dropdown', value='NA'),
                                html.Button('Convert', id='convert-scale-button', n_clicks=0),
                            ], style={'width': '10%', 'display': 'inline-block'})
                        ]),

                        html.Br(), html.Br(),
                        
                        html.H5('Extracted Table'),
                        html.Div([dcc.Input(
                                            id='adding-columns-name-new-table',
                                            placeholder='Enter a column name...',
                                            value='',
                                            style={'padding': 10}
                                                ), 
                                html.Button('Add Column', id='adding-columns-button-new-table', n_clicks=0)
                                ], style={'height': 50}),
                        extracted_table,
                        html.Button('Add Row', id='adding-rows-button-new-table', n_clicks=0),

                        html.Br(), html.Br(),
                        html.H5('Dashboard'),
                        html.Div(id='line-container')
                        ])
