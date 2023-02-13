# NLPeople

Installation:
- pip install dash
- pip install pandas
- pip install opencv-contrib-python
- pip install pytesseract
- pip install CurrencyConverter

Usage:
open terminal, run 'python app.py'

Current App Functionality:
- upload PDF/image function
- output table display
- editable table cells function
- add/delete rows/columns function
- export to xlsx function

What we need to continue implementing:
- highlight metrics that user wants
- extract those metrics only
- currency converter - user to input currency

Flaws of the app:
- cannot rearrange columns or rows
- columns and rows can only be added to the ends of the table
- for currency conversion, need to read the file for whether the values are in billions/millions