#!/usr/bin/env python3

# Detects text in a document stored in an S3 bucket. 
import boto3
import sys
from time import sleep
import math
import pandas as pd
from trp import Document
from PIL import Image, ImageDraw


if __name__ == "__main__":

    def get_rows_columns_map(table_result, blocks_map):
        rows = {}
        for relationship in table_result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    cell = blocks_map[child_id]
                    if cell['BlockType'] == 'CELL':
                        row_index = cell['RowIndex']
                        col_index = cell['ColumnIndex']
                        if row_index not in rows:
                            # create new row
                            rows[row_index] = {}
                            
                        # get the text value
                        rows[row_index][col_index] = get_text(cell, blocks_map)
        return rows


    def get_text(result, blocks_map):
        text = ''
        if 'Relationships' in result:
            for relationship in result['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        word = blocks_map[child_id]
                        if word['BlockType'] == 'WORD':
                            text += word['Text'] + ' '
                        if word['BlockType'] == 'SELECTION_ELEMENT':
                            if word['SelectionStatus'] =='SELECTED':
                                text +=  'X '    
        return text

    def generate_table_csv(table_result, blocks_map, table_index):
        rows = get_rows_columns_map(table_result, blocks_map)

        table_id = 'Table_' + str(table_index)
    
        # get cells.
        csv = 'Table: {0}\n\n'.format(table_id)

        for row_index, cols in rows.items():
        
            for col_index, text in cols.items():
                csv += '{}'.format(text) + ","
            csv += '\n'
        
        csv += '\n\n\n'
        return csv
    
    bucket='fyptestingv2'
    ACCESS_KEY='AKIA4H3NN7J7QNORNKVF'
    SECRET_KEY='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'
    
    textract = boto3.client('textract', 
                        region_name='us-east-1', 
                        aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY)
    
    s3 = boto3.client('s3',  
                    region_name='us-east-1',
                    aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY)


    objs = s3.list_objects(Bucket='fyptestingv2', Delimiter='/')['Contents']
    print (objs)
    objs.sort(key=lambda e:['LastModified'], reverse=True)
    print ("************")
    first_item = list(objs[0].items())[0]
    print(first_item[1])
    documentName = str(first_item[1])
    
    with open(documentName, 'rb') as document:
        response = textract.analyze_document(
            Document={
                
                'Bytes' : document.read(),
            },
            FeatureTypes=['TABLES']
        )
    
    # blocks=response['Blocks']
    # print(blocks)
    blocks=response['Blocks']
    
    #Get the text blocks
    blocks=response['Blocks']

    # Create image showing bounding box/polygon the detected lines/text
    
    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            print ('1')
            table_blocks.append(block)

    if len(table_blocks) <= 0:
        print ("<b> NO Table FOUND </b>")

    csv = ''
    for index, table in enumerate(table_blocks):
        csv += generate_table_csv(table, blocks_map, index +1)
        csv += '\n\n'

    output_file = 'output.csv'

    # replace content
    with open(output_file, "wt") as fout:
        fout.write(csv)

    # show the results
    print('CSV OUTPUT FILE: ', output_file)
    
    
