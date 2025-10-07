# Facial verification with ***Siamese Network***

> [!NOTE]
> Please make sure that you already install python in your computer or raspberry pi.

## Setup a virtual environment

### Walk Into Project Directory

`cd Facial-Verification-with-a-Siamese-Network-`

### Create an Environment

`python3 -m venv .venv`

### Activate an environment

> macOs/linux

`source .venv/bin/activate`

> window

```
env/Scripts/activate.bat //In CMD
env/Scripts/Activate.ps1 //In Powershel
````

### Install All Dependencies We Need !

```
  pip install tensorflow
  pip install firebase_admin
```

or

`pip install -r requirements.txt`

## After You Create An Environment Successful

1. you need to get firebase credential from firebase website.Then you put your credential in **smart-home-cred.json**.
2. after you need to create telegram bot by using bot-father in telegram and then put your bot api to **ultis/constant.py** on variable name TOKEN and USER_ID


## Run Program

> Note: --camera Use To Open Your Own Camera That Available On Your Computer Input

`python3 main.py --camera 0`









