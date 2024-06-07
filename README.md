# Setting up the project

1. Create a virtual environment in the directory of the project.
```
python3 -m venv venv
```

2. Activate the virtual environment on your operating system.

On macOS/ Linux:

```
source venv/bin/activate
```

On Windows (cmd):

```
.venv\Scripts\activate.bat
```

On Windows (Powershell):

```
.venv\Scripts\Activate.ps1
```

3. Install the dependencies into your virtual environment.
```
pip install -r requirements.txt
```

4. Create an .env file, and paste the following code into the file, replacing the text with your secrets.
``` 
CLIENT_ID=YOUR_CLIENT_ID
CLIENT_SECRET=YOUR_CLIENT_SECRET
USERNAME=YOUR_USERNAME
PASSWORD=YOUR_PASSWORD
```

5. Build the project by running the following code in your virtual environment.

```
python setup.py sdist bdist_wheel
```

6. Install the project. You can do so by using the name 'FitSentiment', or installing it in editable mode with -e to make changes.

Example: installing the project by its name. 
```
pip install FitSentiment
```

Example: installing the project in editable mode. 
```
pip install -e .
```