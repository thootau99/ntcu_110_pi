import requests
import time
import subprocess
import rclpy
import threading
from tello_msgs.srv import TelloAction


from rclpy.node import Node
from std_msgs.msg import String

SERVERIP = 'https://powerful-bastion-90835.herokuapp.com/'    # local host, just for testing


class Publisher(Node):

    def __init__(self):
        super().__init__('userCommandNode')

        self.publisher = self.create_publisher(String, 'userCommand')
        self.telloCli = self.create_client(TelloAction, 'tello_action')
        self.telloCliRequest = TelloAction.Request()
    def send(self, s):
        self.publisher.publish(msg=s)
        print("published" + s.data)

    def sendToTello(self, s):
        self.telloCliRequest.cmd = s
        self.future = self.telloCli.call_async(self.telloCliRequest)

    def nothing(self):
        return 0



def getArg(pub):
    while True:
        test = String()
        test.data = "123123123"
        pub.send(test)

        pub.sendToTello("rc 0 1 1 1")
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