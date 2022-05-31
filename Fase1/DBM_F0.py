
from "../imgUtils" import *
from imgStruct import * 


imgDict = initStruct("./images")
subImgArray = splitImg(imgDict[0]["obj"], 8)


print("COLOR MOMENTS - CM:")
print("\tmean color moment:")
avgColor = meanCM(subImgArray[0])
print("\tMean: ",avgColor)
stDv = standardDeviationCM(subImgArray[0], avgColor) 
print("\tStDv: ",stDv)
print("\t\t!!! NOTA -> standard deviation troppo alta??")
skewness= skewnessCM(subImgArray[0], avgColor)
print("\tSkewness: ",skewness)

print("EXTENDED LOCAL BINARY PATTERNS - ELBP:")
elbp = ELBP(subImgArray[0])
print("ELBP matrice di max - min:  (size= ",len(elbp),")")
print(elbp)