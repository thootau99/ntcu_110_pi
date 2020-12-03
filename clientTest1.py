import requests
import time
import subprocess
import rclpy
import threading
import json
from tello_msgs.srv import TelloAction
from threading import BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor
sem = threading.Semaphore()
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()
from rclpy.node import Node
from std_msgs.msg import String

SERVERIP = 'https://xiang.shirinmi.io'    # local host, just for testing 
class BoundedExecutor:
    """BoundedExecutor behaves as a ThreadPoolExecutor which will block on
    calls to submit() once the limit given as "bound" work items are queued for
    execution.
    :param bound: Integer - the maximum number of items in the work queue
    :param max_workers: Integer - the size of the thread pool
    """
    def __init__(self, bound, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = BoundedSemaphore(bound + max_workers)

    """See concurrent.futures.Executor#submit"""
    def submit(self, fn, *args, **kwargs):
        self.semaphore.acquire()
        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except:
            self.semaphore.release()
            raise
        else:
            future.add_done_callback(lambda x: self.semaphore.release())
            return future

    """See concurrent.futures.Executor#shutdown"""
    def shutdown(self, wait=True):
        self.executor.shutdown(wait)

def countdown(t, pub): 
    while t: 
        mins, secs = divmod(t, 60) 
        timer = '{:02d}:{:02d}'.format(mins, secs) 
        print(timer, end="\r") 
        time.sleep(1) 
        t -= 1
    pub.sendToTello("takeoff")
    print('count complete.')
    return True
  
  
exe = BoundedExecutor(4, 100)

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
        r = requests.get(url=SERVERIP + "/set_face?facename=" + resultJson, verify=False)
          
    def setFaceName(self, name):
        n = String()
        n.data = name
        self.publisher.publish(n)
def getArg(pub):
    while True:
    
        r = requests.get(url=SERVERIP + "/update_data", verify=False)
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
            if data['instruction'] == 'timer':
                print('timer', data['min'])
                m = int(data['min']) * 5
                exe.submit(countdown, m, pub)
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
