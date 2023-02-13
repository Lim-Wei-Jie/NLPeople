# NLPeople

To create new env

```
conda env create -f ./environment/conda.yaml
```

To activate env
```
conda activate dash-app
```

To deactivate env
```
conda deactivate
```

To install pip dependencies
```
python -m pip install -r ./environment/requirements.txt
```

Usage:
open terminal, run
```
python app.py
```

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
