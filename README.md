# NLPeople

https://finextract.onrender.com/

For local use, follow the instructions below

To create new env, run CMD as administrator and install virtualenv

```
pip install virtualenv
```

Check if virtualenv is installed
```
virtualenv --version
```

Create env (virtual env name = venv)
```
virtualenv venv
```

Activate virtualenv
```
venv\Scripts\activate
```

Deactivate virtualenv
```
deactivate
```

To install pip dependencies
```
python -m pip install -r ./requirements.txt
```

Usage:
open terminal, run
```
python app.py
```

Flaws of the app:
- cannot rearrange columns or rows
- columns and rows can only be added to the ends of the table
- for currency conversion, need to read the file for whether the values are in billions/millions
