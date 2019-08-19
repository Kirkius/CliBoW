import os
import threading
import _pickle as cpickle
import SaveLoad_Functions.loadfun as lf

class myThread (threading.Thread):
    def __init__(self, threadID, category):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.category = category
    def run(self):
        print ("Starting " + str(self.category))
        # Get lock to synchronize threads
        threadLock.acquire()
        data = lf.loadData() #BROKEN, NEEDS CATEGORY!!
        with open('G:\\Casper\\GitKraken\\srp\\Saves\\Pickle\\{0}'.format(self.category), 'wb') as picklefile:
            cpickle.dump(data, picklefile)
        # Free lock to release next thread
        print('Release category: {}'.format(self.category))
        threadLock.release()

threadLock = threading.Lock()
threads = []
i = 0

for roots, dirs, files in os.walk('G:\\Casper\\GitKraken\\srp\\data\\data(full)'):
    for name in dirs:
        i += 1
        
        # Create new threads
        thread1 = myThread(i, name)

        # Start new Threads
        thread1.start()

        # Add threads to thread list
        threads.append(thread1)

# Wait for all threads to complete
for t in threads:
    t.join()
print ("Exiting Main Thread")