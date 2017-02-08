import os
import csv
import numpy as np
import collections

data = {}      
data2 = {}

def merge(fileDir,outFile,dir1,dir2):
    getdata(fileDir)
    write(outFile,dir1,dir2)

def getdata(fileDir):
    for root, dirs, files in os.walk(fileDir):
        for aFile in files:
            if aFile.endswith(".csv"):
                aFile = os.path.join(root, aFile)
                helper(aFile)

def helper(aFile):
    with open(aFile, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in list(csv_reader)[1::]:
                key=row[0]
                value=int(row[1])
                data.setdefault(key,[]).append(value) 
            

def write(outFile,dir1,dir2):
    for key in data:
        a = np.array(data[key])
        counts = np.bincount(a)
        data2[key] = np.argmax(counts)
    print(data2)
    data3 = {int(old_key): val for old_key, val in data2.items()}
    sorted_data = collections.OrderedDict(sorted(data3.items(), key=lambda t: t[0]))
    print(sorted_data)
    with open(outFile, 'w') as csvfile:
        fieldnames = [dir1, dir2]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in sorted_data.items():
            writer.writerow({dir1: key, dir2: int(value)})

merge('/Users/yanzheng/Desktop/optimize test','final_result.csv','ImageId','Label')
