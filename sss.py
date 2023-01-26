import os
import time

def task(Nd):
    print('Run task %s (%s)...' % (Nd, os.getpid()))
    start = time.time()
    file = "./tasks.py"
    cmd = "python %s %s > ./log/%s.log 2>&1" % (file,Nd,Nd)
    print(cmd)
    os.system(cmd)
    end = time.time()
    print('Task %s(%s) runs %0.2f seconds.' % (Nd,os.getpid(), (end - start)))