import os
import tabula
import requests
import re
from urllib.parse import parse_qsl, urljoin, urlparse
import os
import pandas as pd

path_of_the_directory= 'C:/Users/arkgn/Desktop/GG1'
print("Files and directories in a specified path:")
for filename in os.listdir(path_of_the_directory):
    f = os.path.join(path_of_the_directory,filename)
    if os.path.isfile(f):
        print(f)
        tables = tabula.read_pdf(f, pages="all", multiple_tables=True,encoding='cp1252')
        filename = f.replace(".pdf", "")
        tabula.convert_into(f, '{}.csv'.format(filename), pages='all')
        print(tables)