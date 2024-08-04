import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

from utils.constant import *
import json

cred = credentials.Certificate(SMART_HOME_JSON_FILE)
firebase_admin.initialize_app(cred,{
    'databaseURL' : FIREBASE_URL
})

def update_door(state:int) -> None:
    db.reference().child(ROOT_ACTUATOR_PATH + FRONT_DOOR_PATH).set(state)