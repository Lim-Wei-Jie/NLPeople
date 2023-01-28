#!/usr/bin/env python3

# Detects text in a document stored in an S3 bucket. 
import boto3
import sys
from time import sleep
import math
import pandas as pd


if __name__ == "__main__":

    bucket='fyptestingv2'
    ACCESS_KEY='AKIA4H3NN7J7QNORNKVF'
    SECRET_KEY='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'
    
    client = boto3.client('textract', 
                          region_name='us-east-1', 
                          aws_access_key_id=ACCESS_KEY,
                          aws_secret_access_key=SECRET_KEY)
    
    s3 = boto3.client('s3',  
                      region_name='us-east-1',
                      aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    
    #your_bucket = s3.Bucket(bucket)

    objs = s3.list_objects(Bucket='fyptestingv2', Delimiter='/')['Contents']
    print (objs)
    # objs.sort(key=lambda e:['LastModified'], reverse=True)
    # print ("************")
    # first_item = list(objs[0].items())[0]
    # print(first_item[1])
    extracted_data = []
    for s3_file in objs:
        print(s3_file)
        
        # use textract to process s3 file
        response = client.detect_document_text(
            Document={'S3Object': {'Bucket': bucket, 'Name': s3_file['Key']}})
        
        blocks=response['Blocks']

        for block in blocks:
                if block['BlockType'] != 'PAGE':
                    print('Detected: ' + block['Text'])
                    print('Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")
                    
                    # Example case where you want to extract words with #
                    if("#" in block['Text']):
                        words = block['Text'].split()
                        for word in words:
                               if("#" in word):
                                    extracted_data.append({"word" : word, "file" : s3_file.key, "confidence": "{:.2f}".format(block['Confidence']) + "%"})
        
        # sleep 2 seconds to prevent ProvisionedThroughputExceededException
        sleep(2)

    df = pd.DataFrame(extracted_data)
    df = df.drop_duplicates()
    df.to_csv('output.csv')