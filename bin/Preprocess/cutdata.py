import random
import json

def getData():
        with open('NOTEEVENTS_shuffled.json') as json_data:
                data = json.load(json_data)
        return data

def shuffleData(data):
        return random.shuffle(data)

def getSmallData():
        with open('NOTEEVENTS_shuffled.json') as json_data:
                dir(json_data)
                data = []
                for x in range(50):
                        try:
                                data.append(next(json_data))
                        except StopIteration:
                                print("Only made it to %d" % x)
                #data = [next(json_data) for x in range(500)]
        return data

def saveJson(filename, data):
        with open(filename, 'w') as outfile:
                json.dump(data, outfile)



# small_data = {}
# small_data['row_id'] = []
# small_data['subject_id'] = []
# small_data['category'] = []
# small_data['description'] = []
# small_data['text'] = []

# for line in data[:1000]:
#     small_data['row_id'].append(line('ROW_ID'))
#     small_data['subject_id'].append(line('SUBJECT_ID'))
#     small_data['category'].append(line('CATEGORY'))
#     small_data['description'].append(line('DESCRIPTION'))
#     small_data['text'].append(line('TEXT'))