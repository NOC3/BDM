from searchUtils import *


    
imgDict = initStruct("./images")
finalStruct = computeFeatures(imgDict)

ID = 2
k = 4


print("Cerco i "+str(k)+" elementi più simili a ["+str(ID)+"]\n")
resSearch = searchAll(finalStruct, ID, k)
print("Risultati:")
print(resSearch[0])

for e in resSearch[1]:
    print(e)


showMoreImg(finalStruct)