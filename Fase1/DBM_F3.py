from searchUtils import *


    
imgDict = initStruct("./images")
finalStruct = computeFeatures(imgDict)


ID = 2
method= 2
model = ["moments","ELBP","HOG"]
k = 4

print("Cerco i "+str(k)+" elementi più simili a ["+str(ID)+"]\n")
resSearch = search(finalStruct, ID, model[method], k)
print(resSearch)
showMoreImg(finalStruct)