    # documentName = str('q4fy2022-financial-data.pdf')
    
    # with open(documentName, 'rb') as document:
    #     response = textract.analyze_document(
    #         Document={
                
    #             'Bytes' : document.read(),
    #         },
    #         FeatureTypes=['TABLES', 'FORMS']
    #     )
    
    # # blocks=response['Blocks']
    # # print(blocks)
    # blocks=response['Blocks']
    
    # #Get the text blocks
    # blocks=response['Blocks']

    # # Create image showing bounding box/polygon the detected lines/text
    
    # blocks_map = {}
    # table_blocks = []
    # for block in blocks:
    #     blocks_map[block['Id']] = block
    #     if block['BlockType'] == "TABLE":
    #         print ('1')
    #         table_blocks.append(block)

    # if len(table_blocks) <= 0:
    #     print ("<b> NO Table FOUND </b>")

    # csv = ''
    # for index, table in enumerate(table_blocks):
    #     csv += generate_table_csv(table, blocks_map, index +1)
    #     csv += '\n\n'

    # output_file = 'output.csv'

    # # replace content
    # with open(output_file, "wt") as fout:
    #     fout.write(csv)

    # # show the results
    # print('CSV OUTPUT FILE: ', output_file)