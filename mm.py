import ros_all
import threading
import time

ros = ros_all
t = threading.Thread(target=ros.main)

t.start()

def setFollowName(name):
    ros.MinimalSubscriber.setFollowName(self=ros.MinimalSubscriber, s=name)

def getFollowName():
    return ros.MinimalSubscriber.getFollowName(self=ros.MinimalSubscriber)

def getFaceName():
    return ros.MinimalSubscriber.getFaceNames(self=ros.MinimalSubscriber)

while True:
    print(getFollowName())
    time.sleep(0.3)

