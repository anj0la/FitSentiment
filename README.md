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

5. Add the system path of the parent project directory to your PYTHONPATH in your shell profile. Zsh is typically ~/.zshrc, while Bash is typically ~/.bash_profile or ~/.bashrc.
Replace "~/.zshrc" with the equivalent bash commands.

### ~/.zshrc example

Open the shell profile.
```
sudo nano ~/.zshrc
```

Add the directory to your PYTHONPATH.

```
export PYTHONPATH="/path/to/your/project"
```

Save the file and exit the text editor. Then, run source ~/.zshrc to apply the change.

```
source ~/.zshrc
```
