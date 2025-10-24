In order to run the code, you need to create an environment. To do that, run the following command:
python -m venv [environment_name]

Once the environment is created, activate it using:
.\[environment_name]\Scripts\activate

In case that this fails, run the following commands:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
pip install --upgrade pip
pip install virtualenv
pip install -r requirements.txt

Then, to start the web execute:
python web_app.py

Finally, you will be able to open the web by accessing the following URL at the browser:
http://127.0.0.1:8088/