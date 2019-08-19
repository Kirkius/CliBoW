import pandas
import os
import re
import time

def runset(section = -1): 
    
    data = pandas.read_json('NOTEEVENTS.json')
    x = 0
    y = 0
    print(data[:20])
    #Make folders
    # for category in data['CATEGORY']
    #   category = re.sub(r"/", "_", category)
    #   category = re.sub(r"^\s$", "", category)
    #   data['CATEGORY'][category] = category

    #setup loop
    totalTimeFolder = 0
    totalTimeFile = 0
    start = 0
    n_iter = len(data['CATEGORY'])
    if(section == 1): 
        n_iter = int(n_iter / 2)
        print('running section 1')
    if(section == 2): 
        start = int(n_iter / 2)
        print('running section 2')

    for i in range(start, n_iter):
        start = time.time()
        category = data['CATEGORY'][i]
        category = category.replace('/', '_')
        data['CATEGORY'][i] = category.strip()
        #print(category)
        path = 'data/{}'.format(category)
        if not os.path.exists(path):
            os.mkdir(path)
            x += 1
        end = time.time()
        passedTime = end - start   
        totalTimeFolder = totalTimeFolder + passedTime
        avgTime = totalTimeFolder / (i + 1)
        remainingTime = (avgTime * float(n_iter)) - totalTimeFolder
        print("Processed {0} / {1} files for folders. Estimated time left: {4} Seconds. Average time: {3}s. Read time: {2}s."
            .format(
                i+1,
                n_iter,
                round(passedTime, 4),
                round(avgTime, 2),
                int(round(remainingTime, 0))),
            end="          \r"),
    print("")

    print('Created {} new directories! Time spent: {} seconds'.format(x, round(totalTimeFolder, 2)))

    #Make text files
    start = 0
    n_iter = len(data)
    if(section == 1): 
        n_iter = int(n_iter / 2)
        print('running section 1')
    if(section == 2): 
        start = int(n_iter / 2)
        print('running section 2')
    for i in range(start, n_iter):
        start = time.time()
        path = 'data/{}/{}'.format(data['CATEGORY'][i], data['ROW_ID'][i])
        if not os.path.exists(path):
            with open('data/{}/{}.txt'.format(data['CATEGORY'][i], data['ROW_ID'][i]), 'w') as file:
                file.write(data['TEXT'][i])
            y += 1
        end = time.time()
        passedTime = end - start   
        totalTimeFile = totalTimeFile + passedTime
        avgTime = totalTimeFile / (i + 1)
        remainingTime = (avgTime * float(n_iter)) - totalTimeFile
        print("Created {0} / {1} files. Estimated time left: {4} Seconds. Average time: {3}s. Read time: {2}s."
            .format(
                i+1,
                n_iter,
                round(passedTime, 4),
                round(avgTime, 2),
                int(round(remainingTime, 0))),
            end="          \r"),
    print("")


    print('Created {} new text files! Time spent: {} seconds'.format(y, round(totalTimeFile, 2)))