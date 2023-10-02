from datetime import datetime

def timestamp():
    presentDate = datetime.now()
    unix_timestamp = datetime.timestamp(presentDate)*1000
    return str(int(unix_timestamp))

