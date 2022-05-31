from imgUtils import *
from imgStruct import * 

imgDict = initStruct("./images")



res = computeMoments(imgDict[0])
print(res)