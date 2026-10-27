
===
#how to run venv in vscode:
# How to create venv (one time job)
#1) View -> Command Palette -> "Terminal: Create New Terminal"
#2) python -m venv .venv
# How to activate venv
#3) Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#4) .\.venv\Scripts\activate

----
Ubuntu:
apt install python3.12-venv
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

gunicorn --bind=0.0.0.0:8000 app:app

~~~ GIT ~~~

===
