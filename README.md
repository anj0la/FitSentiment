# Setting up the project (macOS / Linux)

1. Create a virtual environment
```
python3 -m venv venv
```

2. Activate the virtual environment
```
source venv/bin/activate
```

3. Install dependencies
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