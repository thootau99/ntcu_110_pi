import requests
import time
import subprocess
SERVERIP = 'https://powerful-bastion-90835.herokuapp.com/'    # local host, just for testing


def main():
    while True:
        r = requests.get(SERVERIP + "/update_data")
        try:
            data = r.json()
            if data['instruction'] == 'takeoff':
                print('takeoff')
                cmd = 'echo takoff success'
                stat = subprocess.call(cmd,shell=True,executable='/bin/zsh')
            if data['instruction'] == 'land':
                print('land')
                cmd = 'echo land success'
                stat = subprocess.call(cmd,shell=True,executable='/bin/zsh')
        except:
            print('except')
        time.sleep(0.2)



main()