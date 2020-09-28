import requests
import time
import subprocess
import rclpy
import threading
import json
from tello_msgs.srv import TelloAction


from rclpy.node import Node
from std_msgs.msg import String

SERVERIP = 'http://35.201.162.120:5000'    # local host, just for testing


class Publisher(Node):

    def __init__(self):
        super().__init__('userCommandNode')

        self.create_subscription(String, "facename", self.sendBackFacename, 10)
        self.publisher = self.create_publisher(String, 'facenameset')
        self.telloCli = self.create_client(TelloAction, 'tello_action')
        self.telloCliRequest = TelloAction.Request()
        self.testCount = 0
    def send(self, s):
        self.publisher.publish(msg=s)
        print("published" + s.data)

    def sendToTello(self, s):
        self.telloCliRequest.cmd = s
        self.future = self.telloCli.call_async(self.telloCliRequest)

    def sendBackFacename(self, name):
        if name.data == '':
            return 0
        sp = name.data
        sp = sp.split(' ')
        result = []
        for i in sp:
            i = i.split('_')
            if i[0] in result:
                pass
            else:
                result.append(i[0])
        resultJson = '_'.join(result)
        r = requests.get(SERVERIP + "/set_face?facename=" + resultJson)
          
    def setFaceName(self, name):
        n = String()
        n.data = name
        self.publisher.publish(n)
def getArg(pub):
    while True:
    
        r = requests.get(SERVERIP + "/update_data")
        try:
            data = r.json()
            if data['instruction'] == 'takeoff':
                print('takeoff')
                cmd = 'takeoff'
                pub.sendToTello(cmd)
            if data['instruction'] == 'land':
                print('land')
                cmd = 'land'
                pub.sendToTello(cmd)
            if data['instruction'] == 'setname':
                print('setname' + data['name'])
                pub.setFaceName(data['name'])
            if data['instruction'] == 'stop':
                print('stop')
                pub.setFaceName('stop')
            if data['instruction'] == 'start':
                print('start')
                pub.setFaceName('start')
            if data['instruction'] == 'mask':
                print('mask')
                pub.setFaceName('mask')
            if data['instruction'] == 'normal':
                print('normal')
                pub.setFaceName('normal')
        except:
            print('except')
        time.sleep(0.2)

def main(args=None):
    rclpy.init(args=args)

    pub = Publisher()
    t = threading.Thread(target=getArg, args=[pub,])
    t.start()

    rclpy.spin(pub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pub.destroy_node()
    rclpy.shutdown()

main()
