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

import requests

import re
from datetime import datetime

import nltk
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

# initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

dash.register_page(__name__, path='/')

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

        print("LIST OF DFS .records:", pd.concat(list_of_dfs).to_dict('records'))
        # to rearrange row headers and values into the same row
        # example
        # from
        # {'0': Revenue, '1': '', '2': '', '3': ''}
        # {'0': '', '1': 10, '2': 20, '3': 30} 
        # to: {'0': 'Revenue', '1': 10, '2': 20, '3': 30}
        a_list_of_rows_as_dictionaries = pd.concat(list_of_dfs).fillna("").to_dict('records')
        for i in range(len(a_list_of_rows_as_dictionaries)-1):
            if a_list_of_rows_as_dictionaries[i][0] != '' and a_list_of_rows_as_dictionaries[i+1][0]=='':
                switch = False
                for j in range(1, len(a_list_of_rows_as_dictionaries[0])):
                    if a_list_of_rows_as_dictionaries[i][j] == "" and a_list_of_rows_as_dictionaries[i+1][j]!='':
                        switch = True
                    elif a_list_of_rows_as_dictionaries[i][j] == a_list_of_rows_as_dictionaries[i+1][j]:
                        continue
                    elif a_list_of_rows_as_dictionaries[i][j] != "" and a_list_of_rows_as_dictionaries[i+1][j]=='':
                        switch = True
                    else:
                        switch = False
                        exit
                print("SWITCH:", switch)
                # if switch is true, we want to replace current row values with bottom row values
                if switch == True:
                    for k in range(1, len(a_list_of_rows_as_dictionaries[0])):
                        if a_list_of_rows_as_dictionaries[i][k] == '':
                            a_list_of_rows_as_dictionaries[i][k] = a_list_of_rows_as_dictionaries[i+1][k]
                            a_list_of_rows_as_dictionaries[i+1][k] = ''

        # to remove empty rows after shifting
        new_data_to_return = []
        for i in range(len(a_list_of_rows_as_dictionaries)):
            empty_row = True
            for ele in a_list_of_rows_as_dictionaries[i].values():
                if ele != '':
                    empty_row = False
            if empty_row == False:
                new_data_to_return.append(a_list_of_rows_as_dictionaries[i])


        return [{'name': 'Column {}'.format(i), 'id': str(i), 'renamable': True} for i in pd.concat(list_of_dfs).columns], new_data_to_return, contents

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
    elif triggered_id == 'adding-columns-button-new-table' and new_table_data is not None:
        return update_columns_new_table(n_clicks_add_column, new_column_name, existing_columns, new_table_data)
    elif triggered_id == 'adding-rows-button-new-table' and new_table_data is not None:
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
                [Input('pdf-viewer', 'data'),
                Input('spacy-button', 'n_clicks'),
                Input('financial-terms-list', 'value')],
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

@dash.callback([Output('pdf-viewer', 'selected_rows')],
                [Input('financial-terms-cols-boxes', 'value'),
                Input('pdf-viewer', 'data'),
                Input('financial-terms-list', 'value'),
                Input('deselect-rows-button', 'n_clicks')
                ],
                prevent_initial_call = True)
def update_row_selection(value, table_data, metric_list, n_clicks_deselect):
    triggered_id = ctx.triggered_id
    # print(triggered_id)
    if triggered_id == 'deselect-rows-button':
        return deselect_all_rows(n_clicks_deselect)
    elif triggered_id == 'financial-terms-cols-boxes':
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

    if value is not None:  # if user made a radio button selection (chose a financial metric column)
    # if len(value) != 0:
        for i in range(len(table_data)):
            print("da value", value)
            user_selected_value = value[0]
            row_cell = table_data[i][user_selected_value]
            print("myyy i", i)
            print("myyy value", str(value))

            # fill in the empty row cells with possible financial metrics

            # if row_cell == "":
            #     print("went in")
            #     value_position = col_names_list.index(user_selected_value)
            #     print("value position", value_position)

            #     new_user_selected_value = value_position+1  #position of the right next cell (of the empty cell)
                
            #     #if column name (position) is not the last column, we safely select the right next cell
            #     if int(user_selected_value) < len(col_names_list)-1:
            #         right_row_cell = table_data[i][str(new_user_selected_value)]
            #         if right_row_cell != "":
            #             print("right row cell", right_row_cell)
            #             row_cell = right_row_cell

            #     #if this is not the first row (btw i indicates row value), we can safely select the row above
            #     #row above's value is i-1
            #     if i > 0:
            #         up_row_cell = table_data[i-1][user_selected_value]                        
            #         if up_row_cell != "":
            #             print("up row cell", up_row_cell)
            #             row_cell = up_row_cell

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

        # For Rule 1: the first row in the Extracted Table has to be the x-axis hence needs to be filled and classes need to have different namings
        values_in_first_row = dff.iloc[0].to_list()
        x_axis_variable = values_in_first_row[0]
        classes = values_in_first_row[1:]

        # For Rule 2: unique_values_for_rows and current_df_columns are needed to find out whether there are UNIQUE row headers. dashboard can only be displayed without errors when the Extracted Table contains UNIQUE row headers
        # to store all the UNIQUE df.columns values without empty cells
        unique_values_for_rows = []
        for v in new_df.columns:
            if v.strip() != '' and v not in unique_values_for_rows:
                unique_values_for_rows.append(v)
        
        # to store all the df.columns values without empty cells
        current_df_columns = []
        for v in new_df.columns:
            if v.strip() != '':
                current_df_columns.append(v)

        # colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
        #             for i in range(len(new_df))]
        # print("new_df: ", new_df)

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
                ]
            else:
                raise exceptions.PreventUpdate
        else:
            raise exceptions.PreventUpdate
    else:
        raise exceptions.PreventUpdate

@dash.callback(Output('financial-terms-list', 'options'),
                [Input('input-metric', "value"),
                State('financial-terms-list', 'options'),
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



@dash.callback(Output('display-financial-ratio', 'children'),
                [Input('get-financial-ratios', "n_clicks"),
                Input('spacy-button', 'n_clicks'),
                Input('get-new-table-button', 'n_clicks'),
                Input('financial-terms-cols-boxes', 'value'),
                Input('new-table', 'columns'),
                Input('new-table', 'data')
                ],
                prevent_initial_call = True)
def generate_financial_ratios(n_clicks_fin_ratio, n_clicks_fin_col, n_clicks_extracted, value, columns, table_data):
    if (n_clicks_fin_col == 0 or n_clicks_extracted ==0 or value == None) and n_clicks_fin_ratio > 0:
        return "Please click on the 'GET COLUMNS WITH METRICS' button and make a selection. Then, click on the 'EXTRACT TABLE' button."
    
    if n_clicks_fin_col > 0 and n_clicks_extracted > 0 and n_clicks_fin_ratio > 0 and value != None:
        print("tabla", table_data)

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
            "cash flows from operating activities": 10, "cash flows from investing activities": 10, "cash flows from financing activities": 10, 
            "cash flow from": 10, "cash flows from": 10} ] # dictionary for words from cash flow statements
        
        income_total_weight = 0
        balance_total_weight = 0
        cash_total_weight = 0
        financial_data_type = ""

        collect_lowered_fin_terms_from_col = []
        for a_dict in table_data:
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


        if financial_data_type == "income statement":

            # calculating operating margin for INCOME STATEMENT #

            operating_margin_numerator = ""
            operating_margin_denominator = ""
            operating_margin = "Value cannot be generated, please make further edits to extracted table."
            for a_dict in table_data:
                for word in ["operating profit", "income from operations"]:
                    if word in a_dict[value].lower():
                        #if a_dict[str(int(value)+1)] != "": 
                        print("entered line 747")
                        if list(a_dict.values())[int(value)+1] != "":
                            print("a_dict val: ", list(a_dict.values())[int(value)+1])
                            #operating_margin_numerator = a_dict[str(int(value)+1)] 
                            operating_margin_numerator = list(a_dict.values())[int(value)+1]
                            operating_margin_numerator = re.sub('[^0-9.]', '', operating_margin_numerator)
                            print("updated numerator", operating_margin_numerator)
                        elif list(a_dict.values())[int(value)+1] == "":
                            if list(a_dict.values())[int(value)+2] != "":
                                print("a_dict val 3: ", list(a_dict.values())[int(value)+2])
                                #operating_margin_numerator = a_dict[str(int(value)+1)] 
                                operating_margin_numerator = list(a_dict.values())[int(value)+2]
                                operating_margin_numerator = re.sub('[^0-9.]', '', operating_margin_numerator)
                                print("updated numerator", operating_margin_numerator)

                if "Revenue" in a_dict[value] :
                    print("line 726 went in")
                    if list(a_dict.values())[int(value)+1] != "": #check if the column beside financial term is "" else the latest value will be updated
                        print("a_dict val 2: ", list(a_dict.values())[int(value)+1])
                        #operating_margin_denominator = a_dict[str(int(value)+1)] 
                        operating_margin_denominator = list(a_dict.values())[int(value)+1]
                        operating_margin_denominator = re.sub('[^0-9.]', '', operating_margin_denominator)
                        print("updated denominator", operating_margin_denominator)
                    elif list(a_dict.values())[int(value)+1] == "":
                        if list(a_dict.values())[int(value)+2] != "":
                            print("a_dict val 3: ", list(a_dict.values())[int(value)+2])
                            operating_margin_denominator = list(a_dict.values())[int(value)+2]
                            operating_margin_denominator  = re.sub('[^0-9.]', '', operating_margin_denominator )
                            print("updated denominator", operating_margin_denominator )

            if operating_margin_numerator != "" and operating_margin_denominator != "":
                operating_margin = round(float(operating_margin_numerator) / float(operating_margin_denominator),2)


            # calculating gross profit margin for INCOME STATEMENT #

            gross_profit_margin_numerator = ""
            gross_profit_margin_denominator = ""
            gross_profit_margin = "Value cannot be generated, please make further edits to extracted table."
            for a_dict in table_data:
                for word in ["profit for the year", "profit attributable", "of the parent", "attributable to owners", "net income"]:
                    if word in a_dict[value].lower():
                        #if a_dict[str(int(value)+1)] != "": 
                        print("entered line 784")
                        if list(a_dict.values())[int(value)+1] != "":
                            print("a_dict val 784: ", list(a_dict.values())[int(value)+1])
                            #operating_margin_numerator = a_dict[str(int(value)+1)] 
                            gross_profit_margin_numerator = list(a_dict.values())[int(value)+1]
                            gross_profit_margin_numerator = re.sub('[^0-9.]', '', gross_profit_margin_numerator)
                            print("updated numerator 784", gross_profit_margin_numerator)
                        elif list(a_dict.values())[int(value)+1] == "":
                            print("YEAH1")
                            if list(a_dict.values())[int(value)+2] != "":
                                print("YEAH2")
                                print("a_dict val 3 line 784: ", list(a_dict.values())[int(value)+2])
                                gross_profit_margin_numerator = list(a_dict.values())[int(value)+2]
                                gross_profit_margin_numerator = re.sub('[^0-9.]', '', gross_profit_margin_numerator)
                                print("updated numerator line 777", gross_profit_margin_numerator)

                if "Revenue" in a_dict[value] :
                    print("line 726 went in")
                    if list(a_dict.values())[int(value)+1] != "": #check if the column beside financial term is "" else the latest value will be updated
                        print("a_dict val 2: ", list(a_dict.values())[int(value)+1])
                        #operating_margin_denominator = a_dict[str(int(value)+1)] 
                        gross_profit_margin_denominator = list(a_dict.values())[int(value)+1]
                        gross_profit_margin_denominator = re.sub('[^0-9.]', '', gross_profit_margin_denominator)
                        print("updated denominator line 786", gross_profit_margin_denominator)
                    elif list(a_dict.values())[int(value)+1] == "":
                        if list(a_dict.values())[int(value)+2] != "":
                            print("a_dict val 3: ", list(a_dict.values())[int(value)+2])
                            gross_profit_margin_denominator = list(a_dict.values())[int(value)+2]
                            gross_profit_margin_denominator = re.sub('[^0-9.]', '', gross_profit_margin_denominator)
                            print("updated denominator", gross_profit_margin_denominator )

            if gross_profit_margin_numerator != "" and gross_profit_margin_denominator != "":
                gross_profit_margin = round(float(gross_profit_margin_numerator) / float(gross_profit_margin_denominator),2)

            final_output = "Operating Margin: " + str(operating_margin) + " " + "Gross Profit Margin: " + str(gross_profit_margin)

            return final_output

        elif financial_data_type == "balance sheet":

          # calculating current ratio for BALANCE SHEET #

            current_ratio_numerator = ""
            current_ratio_denominator = ""
            current_ratio_margin = "Value cannot be generated, please make further edits to extracted table."
            for a_dict in table_data:
                for word in ["total current asset"]:
                    if word in a_dict[value].lower():
                        #if a_dict[str(int(value)+1)] != "": 
                        print("entered line 784")
                        if list(a_dict.values())[int(value)+1] != "":
                            print("a_dict val 784: ", list(a_dict.values())[int(value)+1])
                            current_ratio_numerator = list(a_dict.values())[int(value)+1]
                            current_ratio_numerator = re.sub('[^0-9.]', '', current_ratio_numerator)
                            print("updated numerator 784", current_ratio_numerator)
                        elif list(a_dict.values())[int(value)+1] == "":
                            if list(a_dict.values())[int(value)+2] != "":
                                print("a_dict val 3 line 812: ", list(a_dict.values())[int(value)+2])
                                current_ratio_numerator = list(a_dict.values())[int(value)+2]
                                current_ratio_numerator = re.sub('[^0-9.]', '', current_ratio_numerator)
                                print("updated numerator line 815", current_ratio_numerator)

                if "total current liabilities" in a_dict[value].lower() :
                    print("line 816 went in")
                    if list(a_dict.values())[int(value)+1] != "": #check if the column beside financial term is "" else the latest value will be updated
                        print("a_dict val 2 line 816: ", list(a_dict.values())[int(value)+1])
                        #operating_margin_denominator = a_dict[str(int(value)+1)] 
                        current_ratio_denominator = list(a_dict.values())[int(value)+1]
                        current_ratio_denominator = re.sub('[^0-9.]', '', current_ratio_denominator)
                        print("updated denominator line 816", current_ratio_denominator)
                    elif list(a_dict.values())[int(value)+1] == "":
                        if list(a_dict.values())[int(value)+2] != "":
                            print("a_dict val 3 line 827: ", list(a_dict.values())[int(value)+2])
                            current_ratio_denominator = list(a_dict.values())[int(value)+2]
                            current_ratio_denominator = re.sub('[^0-9.]', '', current_ratio_denominator)
                            print("updated numerator line 829", current_ratio_denominator)

            if current_ratio_numerator != "" and current_ratio_denominator != "":
                current_ratio_margin = round(float(current_ratio_numerator) / float(current_ratio_denominator),2)


            # final_output = "Current Ratio: " + str(current_ratio_margin)


            # calculating quick ratio for BALANCE SHEET #

            quick_ratio_numerator = ""
            quick_ratio_denominator = ""
            quick_ratio_margin = "Value cannot be generated, please make further edits to extracted table."
            for a_dict in table_data:
                for word in ["total cash", "receivable"]:
                    count_words = 0
                    total_numerator = 0.0
                    if word in a_dict[value].lower():
                        #if a_dict[str(int(value)+1)] != "": 
                        print("entered line 844")
                        if list(a_dict.values())[int(value)+1] != "":
                            if word == "total cash":
                                cash_and_equiv = list(a_dict.values())[int(value)+1]
                                cash_and_equiv = re.sub('[^0-9.]', '', cash_and_equiv)
                                count_words += 1
                            if word == "receivable":
                                accounts_receivable = list(a_dict.values())[int(value)+1]
                                accounts_receivable = re.sub('[^0-9.]', '', accounts_receivable)
                                count_words += 1
                            if count_words == 2:
                                quick_ratio_numerator = float(cash_and_equiv) + float(accounts_receivable)
                                print("updated numerator 784", quick_ratio_numerator)
                        elif list(a_dict.values())[int(value)+1] == "":
                            print("slid in 1")
                            if list(a_dict.values())[int(value)+2] != "":
                                print("slid in 2")

                                if word == "total cash":
                                    print("fire in 1")
                                    cash_and_equiv = list(a_dict.values())[int(value)+2]
                                    cash_and_equiv = re.sub('[^0-9.]', '', cash_and_equiv)
                                    count_words += 1
                                    total_numerator += float(cash_and_equiv)
                                    print("cash_equi line 887", cash_and_equiv)
                                if word == "receivable":
                                    print("fire in 2")
                                    accounts_receivable = list(a_dict.values())[int(value)+2]
                                    accounts_receivable = re.sub('[^0-9.]', '', accounts_receivable)
                                    count_words += 1
                                    total_numerator += float(accounts_receivable)
                                    print("accounts receivable line 892", accounts_receivable)
                                if count_words == 2:
                                    print("fire in 3")
                                    #quick_ratio_numerator = float(cash_and_equiv) + float(accounts_receivable)
                                    print("updated numerator 893", quick_ratio_numerator)

                                #quick_ratio_numerator = list(a_dict.values())[int(value)+2]
                                #quick_ratio_numerator = re.sub('[^0-9.]', '', quick_ratio_numerator)
                                quick_ratio_numerator = total_numerator
                                print("updated numerator line 897", quick_ratio_numerator)

                if "total current liabilities" in a_dict[value].lower() :
                    print("line 816 went in")
                    if list(a_dict.values())[int(value)+1] != "": #check if the column beside financial term is "" else the latest value will be updated
                        print("a_dict val 2 line 816: ", list(a_dict.values())[int(value)+1])
                        #operating_margin_denominator = a_dict[str(int(value)+1)] 
                        quick_ratio_denominator = list(a_dict.values())[int(value)+1]
                        quick_ratio_denominator = re.sub('[^0-9.]', '', quick_ratio_denominator)
                        print("updated denominator line 816", quick_ratio_denominator)
                    elif list(a_dict.values())[int(value)+1] == "":
                        if list(a_dict.values())[int(value)+2] != "":
                            print("a_dict val 3: ", list(a_dict.values())[int(value)+2])
                            quick_ratio_denominator = list(a_dict.values())[int(value)+2]
                            quick_ratio_denominator = re.sub('[^0-9.]', '', quick_ratio_denominator)
                            print("updated denominator", quick_ratio_denominator )

            if quick_ratio_numerator != "" and quick_ratio_denominator != "":
                quick_ratio_margin = round(float(quick_ratio_numerator) / float(quick_ratio_denominator),2)


            final_output = "Current Ratio: " + str(current_ratio_margin) + " " + "Quick Ratio: " + str(quick_ratio_margin)


            return final_output

        # return financial_data_type


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

input_metric = dcc.Input(id='input-metric',
                        type='text',
                        placeholder='financial metric')

financial_terms_list = dcc.Checklist(
                                options=["revenue", "cost", "gross", "profit", "loss", "net", "ebitda",
                                "equity", "asset", "debt", "cash", "liability", "rev", "operation", "receivable"],
                                value=["revenue", "cost", "gross", "profit", "loss", "net", "ebitda",
                                "equity", "asset", "debt", "cash", "liability", "rev", "operation", "receivable"],
                                id='financial-terms-list',
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
                                html.Button('Deselect All', id='deselect-rows-button', n_clicks=0),
                                html.Button('Extract Table', id='get-new-table-button', n_clicks=0),
                                html.P("From the shortlisted columns below, please choose 1 column which contains the financial metrics."),
                                financial_terms_cols_checklist
                                ], style={'width': '42%', 'display': 'inline-block'}),
                            html.Div(style={'width': '2%', 'display': 'inline-block'}),
                            html.Div([
                                html.H5('Metrics'),
                                html.P("All Metrics:"),
                                financial_terms_list,
                                html.P("Add Metric:"),
                                input_metric,
                                html.Button('Add', id='add-metric-button', n_clicks=0),
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
                        html.P("For the dashboard to work, there are a few assumptions:"),
                        html.P("1) The first row in the Extracted Table will be the x-axis of the graph, where the first cell is variable name and the subsequent cells are the UNIQUE classes of the x-axis (the classes have to be in continuous ascending order if they are numbers)"),
                        html.P("2) Extracted Table must only have 1 column of UNIQUE row headers (Must be the first column in the table)"),
                        html.P("For the graph to appear, click any cell in the Extracted Table except for the first row."),
                        html.Div(id='line-container'),

                        html.Br(),

                        html.Div([
                        html.H5('Financial Ratios'),
                        html.Button('Get Financial Ratios!', id='get-financial-ratios', n_clicks=0),
                        html.Br(), html.Br(),
                        html.Div(id='display-financial-ratio'),
                        html.Br(), html.Br(),
                        ])
                        ])