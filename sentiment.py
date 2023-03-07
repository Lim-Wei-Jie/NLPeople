
import camelot as cam
import PyPDF2
from PyPDF2 import PdfReader
import pandas as pd
import re
import nltk
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

# initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


readpdf = PdfReader("input.pdf")
totalpages = len(readpdf.pages)

file_dict = {}

for num_page in range(1, int(totalpages)+1):
    file_dict["page_" + str(num_page)] = cam.read_pdf("input.pdf", flavor='stream', pages=str(num_page), edge_tol=500)
    
list_of_dfs = []
for k,v in file_dict.items():
    for i in range(len(file_dict[k])):
        list_of_dfs.append(file_dict[k][i].df)

# print (list_of_dfs)
print (list_of_dfs[0][0])

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
        # print("SWITCH:", switch)
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
        
# print (range(new_data_to_return))
print (new_data_to_return[1])

bag_of_words = []
for dict in new_data_to_return:
    print(dict)
    number_of_instances_in_col = 0
    for i in range(len(dict)): # each obj in the list represents a row
        row_cell = dict[i] # this is name retrieved for every row
        print('what is row_cell: ', row_cell)
        if row_cell != "" and row_cell != None:
            # print("check row cell type: ", row_cell, type(row_cell), "row num:", i, "col_name:", col_name)
            tokenised_row_cell = row_cell.split(" ")
            for token in tokenised_row_cell:
                #remove special ch
                #convert to lower
                # print("check token:", token, "!!!col name:", col_name)
                clean_token_v1 = re.sub(r'[^a-zA-Z]', '', token)
                clean_token_v2 = clean_token_v1.lower()
                clean_token_v3 = lemmatizer.lemmatize(clean_token_v2)
                print (clean_token_v3)
                print (type(clean_token_v3))
                if clean_token_v3 != "":
                    if clean_token_v3 not in bag_of_words:
                        bag_of_words.append(clean_token_v3)
print (bag_of_words)