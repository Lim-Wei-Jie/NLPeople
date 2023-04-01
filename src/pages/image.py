import base64
import io
import dash
from dash import Dash, dash_table, dcc, html, exceptions, ctx
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

import dash_bootstrap_components as dbc

import cv2
import pytesseract

import requests

import re
from datetime import datetime

import nltk
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

# initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"/usr/local/Cellar/tesseract/5.3.0_1/bin/tesseract"

dash.register_page(__name__)

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
                Input('convert-scale-button', 'n_clicks'),
                Input('scale-input-dropdown', 'value'),
                Input('scale-output-dropdown', 'value'),
                Input('img-viewer', 'selected_cells'),
                State('img-viewer', 'data')],
                prevent_initial_call = True
                )
def update_table(contents, filename, n_clicks, value, existing_columns, n_clicks_row, n_clicks_convert, currency_input, currency_output, n_clicks_scale, scale_input, scale_output, selected_cells, table_data):
    triggered_id = ctx.triggered_id
    # print(triggered_id)
    if triggered_id == 'img-upload':
        return img_output(contents, filename)
    elif triggered_id == 'adding-columns-button' and table_data is not None:
        return update_columns(n_clicks, value, existing_columns, table_data, contents)
    elif triggered_id == 'adding-rows-button' and table_data is not None:
        return add_row(n_clicks_row, table_data, existing_columns, contents)
    elif triggered_id == 'convert-currency-button' and table_data is not None:
        return convert_currency(n_clicks_convert, table_data, selected_cells, currency_input, currency_output, existing_columns, contents)
    elif triggered_id == 'convert-scale-button' and table_data is not None:
        return convert_scale(n_clicks_scale, table_data, selected_cells, scale_input, scale_output, existing_columns, contents)
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
            count = 0
            for i in range(0, len(row.split())-1):
                if re.search("[0-9]", row.split()[i]) and re.search("[0-9]", row.split()[i+1]):
                    count += 1
            lengths_of_each_rows.append(count)

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
                if re.search("[0-9]", row.split()[-1]):
                    numbers.append(row.split()[-1])
                else:
                    if len(numbers) == 0 and row.split()[0] != row.split()[-1]:
                        words.append(row.split()[-1])

            whole = []
            whole.extend(words)

            temp = " ".join(whole)
            whole = [temp]

            numbers_length = len(numbers)
            for i in range(0, max(lengths_of_each_rows)+1 - numbers_length):
                whole.append("")
            whole.extend(numbers)
            data.append(whole)

        df = pd.DataFrame(data)

        return [{'name': 'Column {}'.format(i), 'id': str(i), 'renamable': True} for i in df.columns], df.to_dict('records'), contents


def update_columns(n_clicks, value, existing_columns, table_data, contents):
    existing_column_ids = []
    for i in range(len(existing_columns)):
        existing_column_ids.append(existing_columns[i]['id'])
    
    if n_clicks > 0:
        existing_columns.append({
            'name': value, 'id': str(int(existing_column_ids[-1])+1), 
            'renamable': True
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
        url = "https://api.exchangerate.host/convert?from=" + str(currency_input) + "&to=" + str(currency_output)
        response = requests.get(url)
        data = response.json()
        exchange_rate = data['result'] 
        exchange_date = data['date']
        historical = data['historical'] # check if exchange rate is historical or not (realtime)

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
                        converted_amount = cell_value * exchange_rate
                        table_data[cell['row']][cell['column_id']] = str(converted_amount)
            else:
                # Print the cell value if it's not a number
                print(cell_value) 

        if currency_input != currency_output:
            exchange_rate_time = str(datetime.today())
            if historical == True:
                exchange_rate_time = exchange_date
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
    elif triggered_id == 'adding-columns-button-new-table' and new_table_data is not None:
        return update_columns_new_table(n_clicks_add_column, new_column_name, existing_columns, new_table_data)
    elif triggered_id == 'adding-rows-button-new-table' and new_table_data is not None:
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
    existing_column_ids = []
    for i in range(len(existing_columns)):
        existing_column_ids.append(existing_columns[i]['id'])
    
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

@dash.callback(Output('financial-terms-cols-boxes-img', 'options'),
                [Input('img-viewer', 'data'),
                Input('spacy-button', 'n_clicks'),
                Input('financial-terms-list-img', 'value')],
                prevent_initial_call = True)
def shortlisted_financial_term_columns_for_user_selection(table_data, n_clicks, metric_list):

    if n_clicks > 0 and table_data is not None:
        # stemmed_master_list = ["revenue", "cost", "gross", "profit", "loss", "net", "ebitda",
        #                         "equity", "asset", "debt", "cash", "liability", "rev"]
        
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
                        for keyword in metric_list:
                            if keyword.lower() in clean_token_v3:
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

@dash.callback([Output('img-viewer', 'selected_rows')],
                [Input('financial-terms-cols-boxes-img', 'value'),
                Input('img-viewer', 'data'),
                Input('financial-terms-list-img', 'value'),
                Input('deselect-rows-button', 'n_clicks')
                ],
                prevent_initial_call = True)
def update_row_selection(value, table_data, metric_list, n_clicks_deselect):
    triggered_id = ctx.triggered_id
    # print(triggered_id)
    if triggered_id == 'deselect-rows-button':
        return deselect_all_rows(n_clicks_deselect)
    elif triggered_id == 'financial-terms-cols-boxes-img':
        return shortlisted_financial_term_columns_for_user_selection(value, table_data, metric_list)
    else:
        raise exceptions.PreventUpdate

def deselect_all_rows(n_clicks_deselect):
    if n_clicks_deselect > 0:
        selected_rows = [[]]
    else:
        raise exceptions.PreventUpdate
    return selected_rows

def shortlisted_financial_term_columns_for_user_selection(value, table_data, metric_list):   
    # if len(value) != 0:
    if value is not None:
        print("value", value)
        print("data for shortlisted-fin-terms-cols", table_data)

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

            if row_cell != "" and row_cell != None:
                tokenised_row_cell = row_cell.split(" ")
                for token in tokenised_row_cell:
                    #remove special ch
                    #convert to lower
                    clean_token_v1 = re.sub(r'[^a-zA-Z]', '', token)
                    clean_token_v2 = clean_token_v1.lower()
                    clean_token_v3 = lemmatizer.lemmatize(clean_token_v2)
                    for keyword in metric_list:
                        if keyword.lower() in clean_token_v3:
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

@dash.callback(
    [Output(component_id='line-container-img', component_property='children'),
    Output('rules-msg-img', 'children')],
    [Input(component_id='new-table-img', component_property="derived_virtual_data"),
    Input(component_id='new-table-img', component_property='derived_virtual_selected_rows'),
    Input('new-table-img', 'active_cell')],
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

        # For Rule 1: the first row in the Extracted Table has to be the x-axis hence needs to be filled and classes need to have different namings
        values_in_first_row = dff.iloc[0].to_list()
        x_axis_variable = values_in_first_row[0]
        classes = values_in_first_row[1:]

        # For Rule 2: unique_values_for_rows and current_df_columns are needed to find out whether there are UNIQUE row headers. dashboard can only be displayed without errors when the Extracted Table contains UNIQUE row headers
        # to store all the UNIQUE df.columns values without empty cells
        unique_values_for_rows = []
        for v in new_df.columns:
            if v is None:
                v = ""
            if v.strip() != '' and v not in unique_values_for_rows:
                unique_values_for_rows.append(v)
        
        # to store all the df.columns values without empty cells
        current_df_columns = []
        for v in new_df.columns:
            if v != None:
                if v.strip() != '':
                    current_df_columns.append(v)

        # colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
        #             for i in range(len(new_df))]
        # print("new_df: ", new_df)

        # alert msgs based on the rules
        rules_msg = []
        if len(set(classes))!=len(classes):
            rule1_msg = dbc.Alert("First row is the x-axis. Make classes of x-axis unique!", color="danger")
        else:
            rule1_msg = ""
        if len(unique_values_for_rows) != len(current_df_columns):
            rule2_msg = dbc.Alert("Make row headers unique!", color="danger")
        else:
            rule2_msg = ""
        if x_axis_variable != None:
            if x_axis_variable.strip()=='':
                rule3_msg = dbc.Alert("Fill up x-axis variable in the first cell!", color="danger")
                rules_msg.append(rule3_msg)
            else:
                rule3_msg = ""
                rules_msg.append(rule3_msg)
        if x_axis_variable == None:
                rule3_msg = dbc.Alert("Fill up x-axis variable in the first cell!", color="danger")
                rules_msg.append(rule3_msg)


        rules_msg.append(rule1_msg)
        rules_msg.append(rule2_msg)

        # Show dashboard based on the 2 rules, and datatable cannot be empty
        if active_cell is not None and len(new_df)>1 and len(unique_values_for_rows) == len(current_df_columns) and x_axis_variable.strip()!='' and len(set(classes))==len(classes):
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
                                    x = new_df.columns[0],
                                    y = new_df.columns[row]
                                    # labels = {"Revenue": "% of Pop took online course"}
                                ).update_layout(autotypenumbers='convert types')
                                # ).update_layout(showlegend=False, xaxis={'categoryorder': 'total ascending'})
                                # .update_traces(marker_color=colors, hovertemplate="<b>%{y}%</b><extra></extra>")
                                )
                ], rules_msg
            else:
                raise exceptions.PreventUpdate
        else:
            return "", rules_msg
    else:
        raise exceptions.PreventUpdate

@dash.callback(Output('financial-terms-list-img', 'options'),
                [Input('input-metric', "value"),
                State('financial-terms-list-img', 'options'),
                Input('add-metric-button', 'n_clicks')],
                prevent_initial_call = True)
def update_metrics_list(new_metric, metric_list, n_clicks):
    triggered_id = ctx.triggered_id
    if triggered_id == 'add-metric-button' and new_metric is not None:
        return add_metric_to_list(new_metric, metric_list, n_clicks)
    else:
        raise exceptions.PreventUpdate

def add_metric_to_list(new_metric, metric_list, n_clicks):
    metric_list_in_lower = []
    for metric in metric_list:
        metric_list_in_lower.append(metric.lower())

    if n_clicks > 0:
        if new_metric.lower() not in metric_list_in_lower:
            metric_list.append(new_metric)
    else:
        raise exceptions.PreventUpdate
    return metric_list



@dash.callback(Output('display-financial-ratio-img', 'children'),
                [Input('get-financial-ratios', "n_clicks"),
                Input('spacy-button', 'n_clicks'),
                Input('get-new-table-button', 'n_clicks'),
                Input('financial-terms-cols-boxes-img', 'value'),
                Input('new-table-img', 'columns'),
                Input('new-table-img', 'data'),
                Input('financial-terms-cols-boxes-img', 'options'),
                # Input('pdf-viewer', 'selected_rows')
                ],
                prevent_initial_call = True)
def generate_financial_ratios(n_clicks_fin_ratio, n_clicks_fin_col, n_clicks_extracted, value, columns, table_data, options):

    # if (n_clicks_fin_col == 0 or n_clicks_extracted ==0 or value == None) and n_clicks_fin_ratio > 0:
    #     return "Please click on the 'GET COLUMNS WITH METRICS' button and make a selection. Then, click on the 'EXTRACT TABLE' button."
    
    if (n_clicks_fin_col > 0 and n_clicks_extracted > 0 and n_clicks_fin_ratio > 0 and value != None) or (n_clicks_extracted > 0 and n_clicks_fin_ratio > 0):
        print("tabla", table_data)
        # print("selected_rows", selected_rows)

        print("options", options, type(options))
            #options = {"0":"0", "4","4"}
        # if options != None:
        #     for option in options: #drop unselected possible financial cols in extracted_table
        #         if option != value:
        #             for row in range(len(table_data)):
        #                 if option in table_data[row].keys():
        #                     table_data[row].pop(option)

        if n_clicks_extracted and n_clicks_fin_ratio > 0:
            value = "0"

        


        #Step 1: App determines what kind of financial data table_data is. (implement NLP if needed)
            #go through row cells in selected financial column to find terms 
            #checks if table_data is from an income statement, balance sheet or cash-flow statement.

      
        financial_data_type_list = [ { "revenue": 1, "operating profit": 2, "gross profit": 1, "earnings per share": 2, "owners of the parent": 1, "cost of": 1,
            "other income": 1, "controlling interest": 1, "net income": 1, "interest expense": 1, "per common share": 2, "per share": 1,
            "tax expense": 1, "sales": 1 }, # dictionary for words from income statements

        { "current assets": 2, "fixed assets": 2, "total assets": 2, "inventory": 1, "inventories": 1, "liabilities": 1, "current liabilities": 3,
            "taxes payable": 1, "long-term debt": 3, "total current liabilities": 3, "equity": 1, "shareholders' equity": 2, "common shares": 2,
            "common stock": 2 }, # dictionary for words from balance sheets
        
        { "cash flow from operating activities": 10, "cash flow from investing activities": 10, "cash flow from financing activities": 10,
            "cash flows from operating activities": 15, "cash flows from investing activities": 10, "cash flows from financing activities": 10, 
            "cash flow from": 10, "cash flows from": 10} ] # dictionary for words from cash flow statements
        
        income_total_weight = 0
        balance_total_weight = 0
        cash_total_weight = 0
        financial_data_type = ""

        collect_lowered_fin_terms_from_col = []
        for a_dict in table_data:
            if a_dict[value] != None:
                collect_lowered_fin_terms_from_col.append(a_dict[value].lower())
        print("collect_lowered_fin_terms_from_col: ", collect_lowered_fin_terms_from_col)

        for fin_data_type_dict in financial_data_type_list:
            if financial_data_type_list[0] == fin_data_type_dict:
                for income_key in fin_data_type_dict:
                    for collected_term in collect_lowered_fin_terms_from_col:
                        if income_key in collected_term:
                            income_total_weight += fin_data_type_dict[income_key]
                            print("added income key: ", income_key)

            if financial_data_type_list[1] == fin_data_type_dict:
                for balance_key in fin_data_type_dict:
                    for collected_term in collect_lowered_fin_terms_from_col:
                        if balance_key in collected_term:
                            balance_total_weight += fin_data_type_dict[balance_key]
                            print("added balance key: ", balance_key)


            if financial_data_type_list[2] == fin_data_type_dict:
                for cash_key in fin_data_type_dict:
                    for collected_term in collect_lowered_fin_terms_from_col:
                        if cash_key in collected_term:
                            cash_total_weight += fin_data_type_dict[cash_key]
                            print("added cash key: ", cash_key)

        list_of_weights = [income_total_weight, balance_total_weight, cash_total_weight]

        print("list_of_weights: ", list_of_weights)
        max_value = max(list_of_weights)
        index = list_of_weights.index(max_value)

        if index == 0:
            financial_data_type = "income statement"
            print("financial data type is income statement")
        elif index == 1:
            financial_data_type = "balance sheet"
            print("financial data type is balance sheet")
        else:
            financial_data_type = "cash flow statement"
            print("financial data type is cash flow statement")

        key_metrics = {}  #{'Operating Margin': {'2020': 0.11, '2021': 0.1}, 'Gross Profit Margin': {'2020': 0.07, '2021': 0.06}}
        card_colors = {} #{'Operating Margin': {'2020': "Danger", '2021': "Danger"}, 'Gross Profit Margin': {'2020': "Success", '2021': "Success"}}
        metrics_used = {} #{'Operating Margin': {'2020': [["", "", ""], [""]], '2021': ["",""]} , 'Gross Profit Margin' : {'2020': [["", ""]], '2021': ["", ""]}} #add each word in the list into a substring
        #metrics_used:  {'Operating Margin': {'2020': [['Operating profit'], ['IAS 1.82(a)\nRevenue']], '2021': [['Operating profit'], ['IAS 1.82(a)\nRevenue']], '2023': [[], ['IAS 1.82(a)\nRevenue']]}, 'Gross Margin': {'2023': [['IAS 1.82(b)\nFinance costs', 'IAS 1.82(a)\nRevenue'], ['IAS 1.82(a)\nRevenue']], '2022': [['IAS 1.82(b)\nFinance costs', 'IAS 1.82(a)\nRevenue'], ['IAS 1.82(a)\nRevenue']], '2021': [['IAS 1.82(b)\nFinance costs', 'IAS 1.82(a)\nRevenue'], ['IAS 1.82(a)\nRevenue']]}}
        financial_ratio_cards = []


        if financial_data_type == "income statement":

            # calculating operating margin for INCOME STATEMENT #

            operating_margin_numerator = ""
            operating_margin_denominator = ""
            operating_margin = "Value cannot be generated, please make further edits to extracted table."

            cost_of_goods_sold_row_op_position = ""
            revenue_row_op_position = ""
            operating_expenses_row_op_position = ""
            word_operating_profit_row_op_position = ""
            gross_profit_position = ""

            cost_of_goods_sold_op = ""
            revenue_attained_op = ""
            operating_expenses = ""
            operating_profit = ""
            gross_profit_attained = ""


            for s in range(len(table_data)):
                if table_data[s]["0"] != None:
                    for word_cost in ["cost of goods sold", "cogs" , "cost of goods", "cost of revenue"]:
                        if word_cost in table_data[s]["0"].lower().strip():
                            cost_of_goods_sold_row_op_position = s
                    for word_rev in ["revenue", "net sales"]:
                        if word_rev in table_data[s]["0"].lower().strip():
                            revenue_row_op_position = s
                    for word_opex in ["operating expense", "operating expenses"]:
                        if word_opex in table_data[s]["0"].lower().strip():
                            operating_expenses_row_op_position  = s
                    for word_operating_profit in ["operating profit", "operating income"]:
                        if word_operating_profit in table_data[s]["0"].lower().strip():
                            word_operating_profit_row_op_position = s
                    for word_rev in ["gross profit"]:
                        if word_rev in table_data[s]["0"].lower().strip():
                            gross_profit_position = s

                            #row want all
                            #col dw want first col for year

            for m in range(len(table_data)):
                a_dict = table_data[m]
                if (cost_of_goods_sold_row_op_position != "" and revenue_row_op_position != "" and operating_expenses_row_op_position != "") or (word_operating_profit_row_op_position != "") or (gross_profit_position != "" and operating_expenses_row_op_position != ""):
                        if a_dict[value] != None:
                            for t in range(1, len(table_data[0])):
                                year = list(table_data[0].values())[t]
                                if a_dict[value] != None:
                                    #for t in range(1,len(table_data[0])): #table_data[0] is an obj that will always contain the years
                                    
                                    if list(a_dict.values())[t] != "" and list(a_dict.values())[t] != None:

                                        if cost_of_goods_sold_row_op_position != "" and revenue_row_op_position != "" and operating_expenses_row_op_position != "":
                                            cost_of_goods_sold_op = list(table_data[cost_of_goods_sold_row_op_position].values())[t]
                                            revenue_attained_op = list(table_data[revenue_row_op_position].values())[t]
                                            operating_expenses = list(table_data[operating_expenses_row_op_position].values())[t]
                                            if cost_of_goods_sold_op != None and revenue_attained_op != None and operating_expenses != None and cost_of_goods_sold_op != "" and  revenue_attained_op != "" and operating_expenses != "":
                                                cost_of_goods_sold_op = re.sub('[^0-9.-]', '', cost_of_goods_sold_op)
                                                revenue_attained_op = re.sub('[^0-9.-]', '', revenue_attained_op)
                                                operating_expenses = re.sub('[^0-9.-]', '', operating_expenses)
                                                numerator_metrics_list = []
                                                numerator_metrics_list.append(list(table_data[cost_of_goods_sold_row_op_position].values())[0])
                                                numerator_metrics_list.append(list(table_data[revenue_row_op_position].values())[0])
                                                numerator_metrics_list.append(list(table_data[operating_expenses_row_op_position].values())[0])

                                                operating_margin_numerator = round(float(revenue_attained_op) - float(cost_of_goods_sold_op) - float(operating_expenses),2)
                                                print("updated operating profit numerator first if", operating_margin_numerator)

                                        elif word_operating_profit_row_op_position != "":
                                            operating_profit = list(table_data[word_operating_profit_row_op_position].values())[t]
                                            print("sweaty monkey: ", operating_profit)
                                            if operating_profit != None and operating_profit != "":
                                                operating_profit = re.sub('[^0-9.-]', '', operating_profit)
                                                numerator_metrics_list = []
                                                numerator_metrics_list.append(list(table_data[word_operating_profit_row_op_position].values())[0])

                                                print("check this operating_profit : ", operating_profit)
                                                operating_margin_numerator = round(float(operating_profit),2)
                                                print("updated numerator first elif", operating_margin_numerator)

                                        elif gross_profit_position != "" and operating_expenses_row_op_position != "":
                                            gross_profit_attained = list(table_data[gross_profit_position].values())[t]
                                            operating_expenses = list(table_data[operating_expenses_row_op_position].values())[t]
                                            if gross_profit_attained != None and operating_expenses != None and gross_profit_attained != "" and operating_expenses != "":
                                                gross_profit_attained = re.sub('[^0-9.-]', '', gross_profit_attained)
                                                operating_expenses = re.sub('[^0-9.-]', '', operating_expenses)
                                                numerator_metrics_list = []
                                                numerator_metrics_list.append(list(table_data[gross_profit_position].values())[0])
                                                numerator_metrics_list.append(list(table_data[operating_expenses_row_op_position].values()[0]))

                                                operating_margin_numerator = round(float(gross_profit_attained) - float(operating_expenses),2)
                                                print("updated operating profit numerator second elif ", operating_margin_numerator)
                                            

                                        for k in range(len(table_data)):
                                            if table_data[k][value] != None:
                                                for word in ["revenue", "net sales"]: 
                                                    if word in table_data[k][value].lower().strip():
                                                        print("gafa firee")
                                                        operating_margin_denominator = list(table_data[k].values())[t]
                                                        if operating_margin_denominator != None and operating_margin_denominator != "":
                                                            operating_margin_denominator = re.sub('[^0-9.-]', '', operating_margin_denominator)
                                                            print("updated denominator", operating_margin_denominator)

                                                            denominator_metrics_list = []
                                                            denominator_metrics_list.append(list(table_data[k].values())[0])

                                                            if operating_margin_numerator != "" and operating_margin_numerator != None and operating_margin_denominator != "" and operating_margin_denominator != None:
                                                                operating_margin = round((float(operating_margin_numerator) / float(operating_margin_denominator))*100,2) # need this line coz for income
                                                                                                                                                                                        #ratios we do subtraction in numerator. If there's "" or None cant compute so error
                                                                print("operating margin check: ", operating_margin)
                                                                if "Operating Margin" not in key_metrics:
                                                                    key_metrics["Operating Margin"] = {}
                                                                    key_metrics["Operating Margin"][year] = str(operating_margin) + "%"
                                                                else:
                                                                    key_metrics["Operating Margin"][year] = str(operating_margin) + "%"
                                                                if "Operating Margin" not in card_colors:
                                                                    card_colors["Operating Margin"] = {}
                                                                    if operating_margin < 20:
                                                                        card_colors["Operating Margin"][year] = "danger"
                                                                    elif operating_margin > 20:
                                                                        card_colors["Operating Margin"][year] = "success"
                                                                else:
                                                                    if operating_margin < 20:
                                                                        card_colors["Operating Margin"][year] = "danger"
                                                                    if operating_margin > 20:
                                                                        card_colors["Operating Margin"][year] = "success"
                                                                    #{'Operating Margin': {'2020': "Danger", '2021': "Danger"}, 'Gross Profit Margin': {'2020': "Success", '2021': "Success"}}
                                                                
                                                                        for i in range(len(key_metrics["Operating Margin"])):
                                                                            if i != 0:
                                                                                if card_colors["Operating Margin"][year] == "success":
                                                                                    if float(str(list(key_metrics["Operating Margin"].values())[i])[:-1]) < float(str(list(key_metrics["Operating Margin"].values())[i-1])[:-1]):
                                                                                        print("current operating margin: ", str(list(key_metrics["Operating Margin"].values())[i])[:-1])
                                                                                        print("before operating margin: ", str(list(key_metrics["Operating Margin"].values())[i-1])[:-1])
                                                                                        list_of_year_keys = list(key_metrics["Operating Margin"].keys()) # aka list of years from key_metrics
                                                                                        low_year = list_of_year_keys[i]
                                                                                        card_colors["Operating Margin"][low_year] = "warning"

                                                                if "Operating Margin" not in metrics_used:
                                                                    metrics_used["Operating Margin"] = {}
                                                                    metrics_used["Operating Margin"][year] = []
                                                                    metrics_used["Operating Margin"][year].append(numerator_metrics_list)
                                                                    metrics_used["Operating Margin"][year].append(denominator_metrics_list)
                                                                else:
                                                                    metrics_used["Operating Margin"][year] = []
                                                                    metrics_used["Operating Margin"][year].append(numerator_metrics_list)
                                                                    metrics_used["Operating Margin"][year].append(denominator_metrics_list)
                                                                
        

            # calculating gross profit margin for INCOME STATEMENT #

            gross_profit_margin_numerator = ""
            gross_profit_margin_denominator = ""
            gross_profit_margin = "Value cannot be generated, please make further edits to extracted table."

            cost_of_goods_sold_row_position = ""
            revenue_row_position = ""
            gross_profit_position = ""

            cost_of_goods_sold = ""
            revenue_attained = ""
            gross_profit_attained = ""

            for s in range(len(table_data)):
                for t in range(1, len(table_data[0])):
                    if table_data[s]["0"] != None:
                        for word_cost in ["cost of goods sold", "cogs" , "cost of goods", "cost of revenue"]:
                            if word_cost in table_data[s]["0"].lower().strip():
                                cost_of_goods_sold_row_position = s
                        for word_rev in ["revenue", "net sales"]:
                            if word_rev in table_data[s]["0"].lower().strip():
                                revenue_row_position = s
                        for word_rev in ["gross profit"]:
                            if word_rev in table_data[s]["0"].lower().strip():
                                gross_profit_position = s

            
            for m in range(len(table_data)):
                a_dict = table_data[m]
                if (cost_of_goods_sold_row_position != "" and revenue_row_position != "") or gross_profit_position != "":
                    if a_dict[value] != None:                        

                        for t in range(1, len(table_data[0])): #table_data[0] is an obj that will always contain the years
                            year = list(table_data[0].values())[t] # don't take table_data[0][0] coz that's the x-axis label --> remember t is index
                            if list(a_dict.values())[t] != "" and list(a_dict.values())[t] != None:

                                if gross_profit_position != "":
                                    gross_profit_attained = list(table_data[gross_profit_position].values())[t]
                                    if gross_profit_attained != None and gross_profit_attained != "":
                                        gross_profit_attained = re.sub('[^0-9.-]', '', gross_profit_attained)

                                        numerator_metrics_list = []
                                        numerator_metrics_list.append(list(table_data[gross_profit_position].values())[0])

                                        gross_profit_margin_numerator = round(float(gross_profit_attained),2)
                                        print("updated gross profit margin numerator if", gross_profit_margin_numerator)


                                elif cost_of_goods_sold_row_position != "" and revenue_row_position != "":

                                    cost_of_goods_sold = list(table_data[cost_of_goods_sold_row_position].values())[t]
                                    revenue_attained = list(table_data[revenue_row_position].values())[t]
                                    if cost_of_goods_sold != None and revenue_attained != None:
                                        cost_of_goods_sold = re.sub('[^0-9.-]', '', cost_of_goods_sold)
                                        print("cost_of_goods for operating_margin", cost_of_goods_sold)
                                        revenue_attained = re.sub('[^0-9.-]', '', revenue_attained)
                                        print("revenue_attained for operating_margin", revenue_attained)

                                        if revenue_attained != "" and cost_of_goods_sold != "":
                                            numerator_metrics_list = []
                                            numerator_metrics_list.append(list(table_data[cost_of_goods_sold_row_position].values())[0])
                                            numerator_metrics_list.append(list(table_data[revenue_row_position].values())[0])
                                            gross_profit_margin_numerator = round(float(revenue_attained) - float(cost_of_goods_sold),2)
                                            print("updated gross profit margin numerator elif", gross_profit_margin_numerator)



                                for k in range(len(table_data)):
                                    if table_data[k][value] != None:
                                        for word in ["revenue", "net sales"]: 
                                            if word in table_data[k][value].lower().strip():
                                                gross_profit_margin_denominator = list(table_data[k].values())[t]
                                                if gross_profit_margin_denominator != None:
                                                    gross_profit_margin_denominator = re.sub('[^0-9.-]', '', gross_profit_margin_denominator)
                                                    print("updated denominator", gross_profit_margin_denominator)

                                                    if gross_profit_margin_denominator != "":
                                                        denominator_metrics_list = []
                                                        denominator_metrics_list.append(list(table_data[k].values())[0])

                                                    if gross_profit_margin_numerator != "" and gross_profit_margin_numerator != None and gross_profit_margin_denominator != "" and gross_profit_margin_denominator != None: # need this line coz for income
                                                                                                                                                                            #ratios we do subtraction in numerator. If there's "" or None cant compute so error
                                                        gross_profit_margin = round((float(gross_profit_margin_numerator) / float(gross_profit_margin_denominator)) * 100,2)
                                                        print("gross profit margin check: ", gross_profit_margin)
                                                        if "Gross Margin" not in key_metrics:
                                                            key_metrics["Gross Margin"] = {}
                                                            string_gross_profit_margin = str(gross_profit_margin) + "%"
                                                            key_metrics["Gross Margin"][year] = string_gross_profit_margin
                                                        else:
                                                            string_gross_profit_margin = str(gross_profit_margin) + "%"
                                                            key_metrics["Gross Margin"][year] = string_gross_profit_margin 
                                                        if "Gross Margin" not in card_colors:
                                                            card_colors["Gross Margin"] = {}
                                                            if gross_profit_margin < 50:
                                                                card_colors["Gross Margin"][year] = "danger"
                                                            elif gross_profit_margin > 50:
                                                                card_colors["Gross Margin"][year] = "success"
                                                        if "Gross Margin" in card_colors:
                                                            if gross_profit_margin < 50:
                                                                card_colors["Gross Margin"][year] = "danger"
                                                            if gross_profit_margin > 50:
                                                                card_colors["Gross Margin"][year] = "success" 
                                                                for i in range(len(key_metrics["Gross Margin"])):
                                                                    if i != 0:
                                                                        if card_colors["Gross Margin"][year] == "success":
                                                                            if float(str(list(key_metrics["Gross Margin"].values())[i])[:-1]) < float(str(list(key_metrics["Gross Margin"].values())[i-1])[:-1]):
                                                                                print("current gross margin: ", str(list(key_metrics["Gross Margin"].values())[i])[:-1])
                                                                                print("before gross margin: ", str(list(key_metrics["Gross Margin"].values())[i-1])[:-1])
                                                                                list_of_year_keys = list(key_metrics["Gross Margin"].keys()) # aka list of years from key_metrics
                                                                                low_year = list_of_year_keys[i]
                                                                                card_colors["Gross Margin"][low_year] = "warning"

                                                        if "Gross Margin" not in metrics_used:
                                                                metrics_used["Gross Margin"] = {}
                                                                metrics_used["Gross Margin"][year] = []
                                                                metrics_used["Gross Margin"][year].append(numerator_metrics_list)
                                                                metrics_used["Gross Margin"][year].append(denominator_metrics_list)
                                                        else:
                                                            metrics_used["Gross Margin"][year] = []
                                                            metrics_used["Gross Margin"][year].append(numerator_metrics_list)
                                                            metrics_used["Gross Margin"][year].append(denominator_metrics_list)

                                                        
            print("key_metrics for income statement: ", key_metrics)
            print("metrics_used: ", metrics_used)
        #metrics_used:  {'Operating Margin': {'2020': [['Operating profit'], ['IAS 1.82(a)\nRevenue']], '2021': [['Operating profit'], ['IAS 1.82(a)\nRevenue']], '2023': [[], ['IAS 1.82(a)\nRevenue']]}, 'Gross Margin': {'2023': [['IAS 1.82(b)\nFinance costs', 'IAS 1.82(a)\nRevenue'], ['IAS 1.82(a)\nRevenue']], '2022': [['IAS 1.82(b)\nFinance costs', 'IAS 1.82(a)\nRevenue'], ['IAS 1.82(a)\nRevenue']], '2021': [['IAS 1.82(b)\nFinance costs', 'IAS 1.82(a)\nRevenue'], ['IAS 1.82(a)\nRevenue']]}}

        
            print("card colors for income statement: " , card_colors)

            for financial_ratio in key_metrics:
                for year_key in key_metrics[financial_ratio]:
                    no_spaces_financial_ratio = financial_ratio.replace(" ", "")
                    element = dbc.Card(
                                dbc.CardBody([
                                    html.P(str(year_key) + " " + str(financial_ratio)),
                                    html.H4(str(key_metrics[financial_ratio][year_key]))
                                ]), className='text-center m-4', color=card_colors[financial_ratio][year_key], inverse=True, id=f"{no_spaces_financial_ratio}-{year_key}")
                    tooltip = dbc.Tooltip(f"NUMERATOR METRICS: {' AND '.join(str(x) for x in metrics_used[financial_ratio][year_key][0])} ; DENOMINATOR METRICS: {' AND '.join(str(x) for x in metrics_used[financial_ratio][year_key][1])}", target=f"{no_spaces_financial_ratio}-{year_key}", placement="top")
                    financial_ratio_cards.append(dbc.Col([element, tooltip]))
            return dbc.Container(dbc.Row(financial_ratio_cards))

                    


        elif financial_data_type == "balance sheet":

          # calculating current ratio for BALANCE SHEET #

            current_ratio_numerator = ""
            current_ratio_denominator = ""
            current_ratio_margin = "Value cannot be generated, please make further edits to extracted table."

            assets_row_position = ""
            liabilities_row_position = ""

            assets_value = ""

            for s in range(len(table_data)):
                for t in range(1, len(table_data[0])):
                    if table_data[s]["0"] != None:
                        for word_cost in ["current assets", "total current assets"]:
                            if word_cost in table_data[s]["0"].lower().strip():
                                assets_row_position = s
                        for word_rev in ["current liabilities"]:
                            if word_rev in table_data[s]["0"].lower().strip():
                                liabilities_row_position = s

            
            for m in range(len(table_data)):
                a_dict = table_data[m]
                if assets_row_position != "" and liabilities_row_position != "":
                    if a_dict[value] != None:
                        for t in range(1, len(table_data[0])): #table_data[0] is an obj that will always contain the years
                            year = list(table_data[0].values())[t] # don't take table_data[0][0] coz that's the x-axis label --> remember t is index
                            if list(a_dict.values())[t] != "" and list(a_dict.values())[t] != None:

                                assets_value = list(table_data[assets_row_position].values())[t]
                                if assets_value != None and assets_value != "":
                                    assets_value= re.sub('[^0-9.-]', '', assets_value)
                                    numerator_metrics_list = []
                                    numerator_metrics_list.append(list(table_data[assets_row_position].values())[0])
                                    current_ratio_numerator = round(float(assets_value),2)
                                    print("updated current ratio numerator", current_ratio_numerator)

                                    for k in range(len(table_data)):
                                        if table_data[k][value] != None:
                                            for word in ["current liabilit"]: 
                                                if word in table_data[k][value].lower().strip():
                                                    current_ratio_denominator = list(table_data[k].values())[t]  
                                                    if current_ratio_denominator!= None and current_ratio_denominator != "":
                                                        current_ratio_denominator = re.sub('[^0-9.-]', '', current_ratio_denominator)
                                                        denominator_metrics_list = []
                                                        denominator_metrics_list.append(list(table_data[k].values())[0])
                                                    print("updated current ratio denominator", current_ratio_denominator)

                                                    if current_ratio_numerator != "" and current_ratio_denominator != "" and current_ratio_numerator != None and current_ratio_denominator != None:
                                                        current_ratio_margin = round(float(current_ratio_numerator) / float(current_ratio_denominator),2)
                                                        print("current ratio check: ", current_ratio_margin)


                                                        if "Current Ratio" not in key_metrics:
                                                            key_metrics["Current Ratio"] = {}
                                                            key_metrics["Current Ratio"][year] = current_ratio_margin
                                                        else:
                                                            key_metrics["Current Ratio"][year] = current_ratio_margin
                                                        if "Current Ratio" not in card_colors:
                                                            card_colors["Current Ratio"] = {}
                                                            if current_ratio_margin < 1.5:
                                                                card_colors["Current Ratio"][year] = "danger"
                                                            elif current_ratio_margin > 1.5:
                                                                card_colors["Current Ratio"][year] = "success"
                                                        else:
                                                            if current_ratio_margin< 1.5:
                                                                card_colors["Current Ratio"][year] = "danger"
                                                            if current_ratio_margin > 1.5:
                                                                card_colors["Current Ratio"][year] = "success"
                                                                for i in range(len(key_metrics["Current Ratio"])):
                                                                    if i != 0:
                                                                        if card_colors["Current Ratio"][year] == "success":
                                                                            if float(str(list(key_metrics["Current Ratio"].values())[i])[:-1]) < float(str(list(key_metrics["Current Ratio"].values())[i-1])[:-1]):
                                                                                list_of_year_keys = list(key_metrics["Current Ratio"].keys()) # aka list of years from key_metrics
                                                                                low_year = list_of_year_keys[i]
                                                                                card_colors["Current Ratio"][low_year] = "warning"
                                                        if "Current Ratio" not in metrics_used:
                                                            metrics_used["Current Ratio"] = {}
                                                            metrics_used["Current Ratio"][year] = []
                                                            metrics_used["Current Ratio"][year].append(numerator_metrics_list)
                                                            metrics_used["Current Ratio"][year].append(denominator_metrics_list)
                                                        else:
                                                            metrics_used["Current Ratio"][year] = []
                                                            metrics_used["Current Ratio"][year].append(numerator_metrics_list)
                                                            metrics_used["Current Ratio"][year].append(denominator_metrics_list)

            print("key_metrics to check current ratio: ", key_metrics)
        

            # calculating quick ratio for BALANCE SHEET #


            quick_ratio_numerator = ""
            quick_ratio_denominator = ""
            quick_ratio_margin = "Value cannot be generated, please make further edits to extracted table."

            current_assets_row_position = ""
            inventory_row_position = ""
            current_liabilities_row_position = ""

            current_assets_value = ""
            inventory_value = ""


            for s in range(len(table_data)):
                for t in range(1, len(table_data[0])):
                    if table_data[s]["0"] != None:
                        for word_cost in ["current assets", "total current assets"]:
                            if word_cost in table_data[s]["0"].lower().strip():
                                current_assets_row_position = s
                        for word_rev in ["inventories"]:
                            if word_rev in table_data[s]["0"].lower().strip():
                                inventory_row_position = s
                        for word_opex in ["current liabilities"]:
                            if word_opex in table_data[s]["0"].lower().strip():
                                current_liabilities_row_position = s


            for m in range(len(table_data)):
                a_dict = table_data[m]
                if current_assets_row_position != "" and inventory_row_position != "" and current_liabilities_row_position != "":
                    print("went into quick ratio checkpoint 0")
                    if a_dict[value] != None:
                        for t in range(1, len(table_data[0])): #table_data[0] is an obj that will always contain the years
                            year = list(table_data[0].values())[t] # don't take table_data[0][0] coz that's the x-axis label --> remember t is index
                            print("went into quick ratio checkpoint 1")
                            if list(a_dict.values())[t] != "" and list(a_dict.values())[t] != None:
                                print("went into quick ratio checkpoint 2")
                                
                                if current_assets_row_position != "" and inventory_value != "":
                                    current_assets_value = list(table_data[current_assets_row_position].values())[t]
                                    inventory_value = list(table_data[inventory_row_position].values())[t]
                                    if current_assets_value != None and inventory_value != None:
                                        current_assets_value = re.sub('[^0-9.-]', '', current_assets_value)            
                                        inventory_value = re.sub('[^0-9.-]', '', inventory_value)
                                
                                        print("current_assets_value", current_assets_value)
                                        print("inventory_value", inventory_value)


                                        if current_assets_value != "" and inventory_value != "":
                                            numerator_metrics_list = []
                                            numerator_metrics_list.append(list(table_data[current_assets_row_position].values())[0])
                                            numerator_metrics_list.append(list(table_data[inventory_row_position].values())[0])

                                            quick_ratio_numerator = round(float(current_assets_value) - float(inventory_value),2)
                                            print("updated quick ratio numerator", quick_ratio_numerator)

                                            for k in range(len(table_data)):
                                                if table_data[k][value] != None:
                                                    for word in ["current liabilities"]:
                                                        if word in table_data[k][value].lower().strip():
                                                            quick_ratio_denominator = list(table_data[k].values())[t]
                                                            if quick_ratio_denominator != None:
                                                                quick_ratio_denominator = re.sub('[^0-9.-]', '', quick_ratio_denominator)
                                                                if quick_ratio_denominator != "":
                                                                    denominator_metrics_list = []
                                                                    denominator_metrics_list.append(list(table_data[k].values())[0])
                                                                    print("updated quick ratio denominator", quick_ratio_denominator)

                                                                if quick_ratio_numerator != "" and quick_ratio_denominator != "":
                                                                    quick_ratio_margin = round(float(quick_ratio_numerator) / float(quick_ratio_denominator),2)
                                                                    print("quick ratio check: ", quick_ratio_margin)


                                                                    if "Quick Ratio" not in key_metrics:
                                                                        key_metrics["Quick Ratio"] = {}
                                                                        key_metrics["Quick Ratio"][year] = quick_ratio_margin
                                                                    else:
                                                                        key_metrics["Quick Ratio"][year] = quick_ratio_margin
                                                                    if "Quick Ratio" not in card_colors:
                                                                        card_colors["Quick Ratio"] = {}
                                                                        if quick_ratio_margin < 1:
                                                                            card_colors["Quick Ratio"][year] = "danger"
                                                                        elif quick_ratio_margin > 1:
                                                                            card_colors["Quick Ratio"][year] = "success"
                                                                    else:
                                                                        if quick_ratio_margin < 1:
                                                                            card_colors["Quick Ratio"][year] = "danger"
                                                                        if quick_ratio_margin > 1:
                                                                            card_colors["Quick Ratio"][year] = "success"
                                                                            for i in range(len(key_metrics["Quick Ratio"])):
                                                                                if i != 0:
                                                                                    if card_colors["Quick Ratio"][year] == "success":
                                                                                        if float(str(list(key_metrics["Quick Ratio"].values())[i])[:-1]) < float(str(list(key_metrics["Quick Ratio"].values())[i-1])[:-1]):
                                                                                            list_of_year_keys = list(key_metrics["Quick Ratio"].keys()) # aka list of years from key_metrics
                                                                                            low_year = list_of_year_keys[i]
                                                                                            card_colors["Quick Ratio"][low_year] = "warning"
                                                                    if "Quick Ratio" not in metrics_used:
                                                                        metrics_used["Quick Ratio"] = {}
                                                                        metrics_used["Quick Ratio"][year] = []
                                                                        metrics_used["Quick Ratio"][year].append(numerator_metrics_list)
                                                                        metrics_used["Quick Ratio"][year].append(denominator_metrics_list)
                                                                    else:
                                                                        metrics_used["Quick Ratio"][year] = []
                                                                        metrics_used["Quick Ratio"][year].append(numerator_metrics_list)
                                                                        metrics_used["Quick Ratio"][year].append(denominator_metrics_list)



            print("key_metrics to check quick ratio: ", key_metrics)

            print("card colors for quick ratio: " , card_colors)

        
            for financial_ratio in key_metrics:
                for year_key in key_metrics[financial_ratio]:
                    no_spaces_financial_ratio = financial_ratio.replace(" ", "")
                    element = dbc.Card(
                                dbc.CardBody([
                                    html.P(str(year_key) + " " + str(financial_ratio)),
                                    html.H4(str(key_metrics[financial_ratio][year_key]))
                                ]), className='text-center m-4', color=card_colors[financial_ratio][year_key], inverse=True, id=f"{no_spaces_financial_ratio}-{year_key}")
                    tooltip = dbc.Tooltip(f"NUMERATOR METRICS: {' AND '.join(str(x) for x in metrics_used[financial_ratio][year_key][0])} ; DENOMINATOR METRICS: {' AND '.join(str(x) for x in metrics_used[financial_ratio][year_key][1])}", target=f"{no_spaces_financial_ratio}-{year_key}", placement="top")
                    financial_ratio_cards.append(dbc.Col([element, tooltip]))
            return dbc.Container(dbc.Row(financial_ratio_cards))


        elif financial_data_type == "cash flow statement":

            # calculating operating cash flow ratio for CASH FLOW STATEMENT #

            operating_cash_flow_numerator = ""
            operating_cash_flow_denominator = ""
            operating_cash_flow_ratio = "Value cannot be generated, please make further edits to extracted table."

            cash_flow_op_row_position = ""
            cash_flow_current_liab_position = ""

            cash_flow_op = ""


            for s in range(len(table_data)):
                for t in range(1, len(table_data[0])):
                    if table_data[s]["0"] != None:
                        for word_cost in ["cash flow from operating activities", "cash flow from operations", "operating cash flow", "cash flows from operating activities", "cash flows from operations"]:
                            if word_cost in table_data[s]["0"].lower().strip():
                                cash_flow_op_row_position = s
                        for word_opex in ["current liabilities"]:
                            if word_opex in table_data[s]["0"].lower().strip():
                                cash_flow_current_liab_position = s


            for m in range(len(table_data)):
                a_dict = table_data[m]
                if cash_flow_op_row_position != "" and cash_flow_current_liab_position != "":
                    if a_dict[value] != None:
                        for t in range(1, len(table_data[0])): #table_data[0] is an obj that will always contain the years
                            year = list(table_data[0].values())[t] # don't take table_data[0][0] coz that's the x-axis label --> remember t is index
                            if list(a_dict.values())[t] != "" and list(a_dict.values())[t] != None:
                                cash_flow_op = list(table_data[cash_flow_op_row_position].values())[t]
                                cash_flow_op = re.sub('[^0-9.-]', '', cash_flow_op)
                                operating_cash_flow_numerator = cash_flow_op

                                if operating_cash_flow_numerator != "":
                                    numerator_metrics_list = []
                                    numerator_metrics_list.append(list(table_data[cash_flow_op_row_position].values())[0])

                                    print("updated operating_cash_flow_numerator", operating_cash_flow_numerator)

                                    for k in range(len(table_data)):
                                        if table_data[k][value] != None:
                                            for word in ["current liabilities"]:
                                                if word in table_data[k][value].lower().strip():
                                                    operating_cash_flow_denominator = list(table_data[k].values())[t]
                                                    operating_cash_flow_denominator = re.sub('[^0-9.-]', '', operating_cash_flow_denominator)
                                                    print("updated operating_cash_flow_denominator", operating_cash_flow_denominator)
                                                    if operating_cash_flow_denominator != "":
                                                        denominator_metrics_list = []
                                                        denominator_metrics_list.append(list(table_data[k].values())[0])

                                                    if operating_cash_flow_denominator != "" and operating_cash_flow_numerator != "":
                                                        operating_cash_flow_ratio = round(float(operating_cash_flow_numerator) / float(operating_cash_flow_denominator),2)
                                                        print("Operating Cash Flow Ratio check: ", operating_cash_flow_ratio)

                                                        if "Operating Cash Flow Ratio" not in key_metrics:
                                                            key_metrics["Operating Cash Flow Ratio"] = {}
                                                            key_metrics["Operating Cash Flow Ratio"][year] = operating_cash_flow_ratio
                                                        else:
                                                            key_metrics["Operating Cash Flow Ratio"][year] = operating_cash_flow_ratio
                                                        if "Operating Cash Flow Ratio" not in card_colors:
                                                            card_colors["Operating Cash Flow Ratio"] = {}
                                                            if operating_cash_flow_ratio < 1:
                                                                card_colors["Operating Cash Flow Ratio"][year] = "danger"
                                                            elif operating_cash_flow_ratio > 1:
                                                                card_colors["Operating Cash Flow Ratio"][year] = "success"
                                                        else:
                                                            if operating_cash_flow_ratio < 1:
                                                                card_colors["Operating Cash Flow Ratio"][year] = "danger"
                                                            if operating_cash_flow_ratio > 1:
                                                                card_colors["Operating Cash Flow Ratio"][year] = "success" 
                                                                for i in range(len(key_metrics["Operating Cash Flow Ratio"])):
                                                                    if i != 0:
                                                                        if card_colors["Operating Cash Flow Ratio"][year] == "success":
                                                                            if float(str(list(key_metrics["Operating Cash Flow Ratio"].values())[i])[:-1]) < float(str(list(key_metrics["Operating Cash Flow Ratio"].values())[i-1])[:-1]):
                                                                                list_of_year_keys = list(key_metrics["Operating Cash Flow Ratio"].keys()) # aka list of years from key_metrics
                                                                                low_year = list_of_year_keys[i]
                                                                                card_colors["Operating Cash Flow Ratio"][low_year] = "warning"
                                                        if "Operating Cash Flow Ratio" not in metrics_used:
                                                            metrics_used["Operating Cash Flow Ratio"] = {}
                                                            metrics_used["Operating Cash Flow Ratio"][year] = []
                                                            metrics_used["Operating Cash Flow Ratio"][year].append(numerator_metrics_list)
                                                            metrics_used["Operating Cash Flow Ratio"][year].append(denominator_metrics_list)
                                                        else:
                                                            metrics_used["Operating Cash Flow Ratio"][year] = []
                                                            metrics_used["Operating Cash Flow Ratio"][year].append(numerator_metrics_list)
                                                            metrics_used["Operating Cash Flow Ratio"][year].append(denominator_metrics_list)

            print("key_metrics to check Operating Cash Flow Ratio: ", key_metrics)

            
            # calculating cash flow coverage ratio for CASH FLOW STATEMENT #

            cash_flow_coverage_numerator = ""
            cash_flow_coverage_denominator = ""
            cash_flow_coverage_ratio = "Value cannot be generated, please make further edits to extracted table."
            for m in range(len(table_data)):
                a_dict = table_data[m]
                for word in ["cash flow from operating activities", "cash flow from operations", "operating cash flow", "cash flows from operating activities", "cash flows from operations"]:
                    if a_dict[value] != None:
                        if word in a_dict[value].lower().strip():
                            for t in range(1, len(table_data[0])): #table_data[0] is an obj that will always contain the years
                                year = list(table_data[0].values())[t] # don't take table_data[0][0] coz that's the x-axis label --> remember t is index
                                if list(a_dict.values())[t] != "" and list(a_dict.values())[t] != None:
                                    cash_flow_coverage_numerator = list(a_dict.values())[t]
                                    cash_flow_coverage_numerator = re.sub('[^0-9.-]', '', cash_flow_coverage_numerator)
                                    if cash_flow_coverage_numerator != "":
                                        numerator_metrics_list = []
                                        numerator_metrics_list.append(list(table_data[m].values())[0])

                                    print("updated cash flow coverage numerator", cash_flow_coverage_numerator)

                                    for k in range(len(table_data)):
                                        if table_data[k][value] != None:
                                            if "total liabilities" in table_data[k][value].lower().strip():
                                                cash_flow_coverage_denominator = list(table_data[k].values())[t]
                                                cash_flow_coverage_denominator = re.sub('[^0-9.-]', '', cash_flow_coverage_denominator)
                                                if cash_flow_coverage_denominator != "":
                                                    denominator_metrics_list = []
                                                    denominator_metrics_list.append(list(table_data[k].values())[0])

                                                print("updated operating_cash_flow_numerator", cash_flow_coverage_denominator)

                                                if cash_flow_coverage_numerator != "" and cash_flow_coverage_denominator != "":
                                                    cash_flow_coverage_ratio = round(float(cash_flow_coverage_numerator) / float(cash_flow_coverage_denominator),2)
                                                    print("cash_flow_coverage_ratio check: ", cash_flow_coverage_ratio)

                                                    if "Cash Flow Coverage Ratio" not in key_metrics:
                                                        key_metrics["Cash Flow Coverage Ratio"] = {}
                                                        key_metrics["Cash Flow Coverage Ratio"][year] = cash_flow_coverage_ratio
                                                    else:
                                                        key_metrics["Cash Flow Coverage Ratio"][year] = cash_flow_coverage_ratio
                                                    if "Cash Flow Coverage Ratio" not in card_colors:
                                                        card_colors["Cash Flow Coverage Ratio"] = {}
                                                        if cash_flow_coverage_ratio < 1:
                                                            card_colors["Cash Flow Coverage Ratio"][year] = "danger"
                                                        elif cash_flow_coverage_ratio > 1:
                                                            card_colors["Cash Flow Coverage Ratio"][year] = "success"
                                                    else:
                                                        if cash_flow_coverage_ratio < 1:
                                                            card_colors["Cash Flow Coverage Ratio"][year] = "danger"
                                                        if cash_flow_coverage_ratio > 1:
                                                            card_colors["Cash Flow Coverage Ratio"][year] = "success"
                                                            for i in range(len(key_metrics["Cash Flow Coverage Ratio"])):
                                                                if i != 0:
                                                                    if card_colors["Cash Flow Coverage Ratio"][year] == "success":
                                                                        if float(str(list(key_metrics["Cash Flow Coverage Ratio"].values())[i])[:-1]) < float(str(list(key_metrics["Cash Flow Coverage Ratio"].values())[i-1])[:-1]):
                                                                            list_of_year_keys = list(key_metrics["Cash Flow Coverage Ratio"].keys()) # aka list of years from key_metrics
                                                                            low_year = list_of_year_keys[i]
                                                                            card_colors["Cash Flow Coverage Ratio"][low_year] = "warning"
                                                    if "Cash Flow Coverage Ratio" not in metrics_used:
                                                        metrics_used["Cash Flow Coverage Ratio"] = {}
                                                        metrics_used["Cash Flow Coverage Ratio"][year] = []
                                                        metrics_used["Cash Flow Coverage Ratio"][year].append(numerator_metrics_list)
                                                        metrics_used["Cash Flow Coverage Ratio"][year].append(denominator_metrics_list)
                                                    else:
                                                        metrics_used["Cash Flow Coverage Ratio"][year] = []
                                                        metrics_used["Cash Flow Coverage Ratio"][year].append(numerator_metrics_list)
                                                        metrics_used["Cash Flow Coverage Ratio"][year].append(denominator_metrics_list)

            print("key_metrics to check Cash Flow Coverage Ratio: ", key_metrics)


            print("card colors for cash flow to net income: " , card_colors)
        
            for financial_ratio in key_metrics:
                for year_key in key_metrics[financial_ratio]:
                    no_spaces_financial_ratio = financial_ratio.replace(" ", "")
                    element = dbc.Card(
                                dbc.CardBody([
                                    html.P(str(year_key) + " " + str(financial_ratio)),
                                    html.H4(str(key_metrics[financial_ratio][year_key]))
                                ]), className='text-center m-4', color=card_colors[financial_ratio][year_key], inverse=True, id=f"{no_spaces_financial_ratio}-{year_key}")
                    tooltip = dbc.Tooltip(f"NUMERATOR METRICS: {' AND '.join(str(x) for x in metrics_used[financial_ratio][year_key][0])} ; DENOMINATOR METRICS: {' AND '.join(str(x) for x in metrics_used[financial_ratio][year_key][1])}", target=f"{no_spaces_financial_ratio}-{year_key}", placement="top")
                    financial_ratio_cards.append(dbc.Col([element, tooltip]))
            return dbc.Container(dbc.Row(financial_ratio_cards))

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
                                # page_action='none',
                                # fixed_rows={'headers': True},
                                # style_table={'height': 500, 'overflowY': 'auto'},
                                # style_header={'overflowY': 'auto'}
                                id='img-viewer',
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

image = html.Img(id='image', width="100%")

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
                                        selected_rows=[],
                                        column_selectable='multi',
                                        style_table={'overflowY': 'auto'},
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
                                    id='financial-terms-cols-boxes-img', 
                                    inline=True
                                    )

input_metric = dcc.Input(id='input-metric',
                        type='text',
                        placeholder='financial metric')

financial_terms_list = dcc.Checklist(
                                options=["revenue", "cost", "gross", "profit", "loss", "net", "ebitda",
                                "equity", "asset", "debt", "cash", "liability", "rev", "operation", "receivable",
                                "expense", "income", "operating", "liabilities", "inventories", "cash flow"],
                                value=["revenue", "cost", "gross", "profit", "loss", "net", "ebitda",
                                "equity", "asset", "debt", "cash", "liability", "rev", "operation", "receivable",
                                "expense", "income", "operating", "liabilities", "inventories", "cash flow"],
                                id='financial-terms-list-img',
                                inline=True
                            )

preview_image_card = dbc.Card([
    dbc.CardBody([
        html.H4("Preview of Image", className="card-title"),
        image
    ])
], style={"height":"100%"})
# color='rgba(33, 175, 191, 0.7)', 

image_output_table_card = dbc.Card([
    dbc.CardBody([
        html.H4("Image Output Table", className="card-title"),
        html.Div(id='currency-is-in-img'),
        html.Div([add_columns, 
                html.Button('Add Column', id='adding-columns-button', n_clicks=0)
                ], style={'height': 50}),
        img_table,
        html.Button('Add Row', id='adding-rows-button', n_clicks=0),
        html.Button('Get Columns with Metrics!', id='spacy-button', n_clicks=0),
        html.Button('Deselect All', id='deselect-rows-button', n_clicks=0),
        html.Button('Extract Table', id='get-new-table-button', n_clicks=0),
        html.P("From the shortlisted columns below, please choose 1 column which contains the financial metrics."),
        financial_terms_cols_checklist
    ])
], style={"height":"100%"})

functions_card = dbc.Card([
    dbc.CardBody([
        html.H5('Metrics'),
        html.P("All Metrics:"),
        financial_terms_list,
        html.P("Add Metric:"),
        input_metric,
        html.Button('Add', id='add-metric-button', n_clicks=0),
        html.Br(), html.Br(),
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
    ])
], style={"height":"100%"})

first_row_cards = dbc.Row(
    [
        dbc.Col(preview_image_card, width=5),
        dbc.Col(image_output_table_card, width=5),
        dbc.Col(functions_card, width=2),
    ]
)

extracted_table_card = dbc.Card([
    dbc.CardBody([
        html.H4('Extracted Table'),
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

        html.Div([
        html.H4('Financial Ratios'),
        html.Button('Get Financial Ratios!', id='get-financial-ratios', n_clicks=0),
        html.Br(), html.Br(),
        html.Div(id='display-financial-ratio-img'),
        html.Br(), html.Br()
        ])
    ])
], style={"height":"100%"})

dashboard_card = dbc.Card([
    dbc.CardBody([
        html.H4('Dashboard'),
        html.Div(id='rules-msg-img'),
        html.Div(id='line-container-img')
    ])
], style={"height":"100%"})

second_row_cards = dbc.Row(
    [
        dbc.Col(extracted_table_card, width=6),
        dbc.Col(dashboard_card, width=6),
    ]
)

#Place into the app
layout = html.Div([html.H4('Convert Image using OpenCV, PyTesseract, Dash'),
                        img_load,
                        html.Br(),
                        first_row_cards,
                        html.Br(), html.Br(),
                        second_row_cards
                        ])
