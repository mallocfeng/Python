print('Hello World')
import sys
# sys.path.append('C:\Yandle\Develop\Auto Testing\lib64')
sys.path.append('../lib64')
import time      
import jkrc

robot = jkrc.RC("10.5.5.100")#返回一个机器人对象