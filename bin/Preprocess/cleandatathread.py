import os
import threading
import _pickle as cpickle
# from .. import SaveLoad_Functions.loadfun as lf
import cleandatafun as cdf
import time

start = time.time()

def loadPickle(model_name):
    with open('G:\\Casper\\GitKraken\\srp\\Saves\\Pickle\\%s' % (model_name), 'rb') as training_model:
        model = cpickle.load(training_model)
    
    return model

class myThread (threading.Thread):
    def __init__(self, threadID, category):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.category = category
    def run(self):
        print ("Starting " + str(self.category))
        # Get lock to synchronize threads
        threadLock.acquire()
        data = loadPickle(self.category)
        clean_data = cdf.nltkClean(data)
        with open ('G:\\Casper\\GitKraken\\srp\\Saves\\Cleaned Pickle\\{0}'.format(self.category), 'wb') as picklefile:
            cpickle.dump(clean_data, picklefile)
        # Free lock to release next thread
        print('Release category: {}'.format(self.category))
        threadLock.release()

threadLock = threading.Lock()
threads = []
i = 0

for roots, dirs, files in os.walk('G:\\Casper\\GitKraken\\srp\\Saves\\Pickle'):
    for name in files:
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
passedTime = time.time() - start
print('Time passed: {} seconds'. format(round(passedTime, 4)))