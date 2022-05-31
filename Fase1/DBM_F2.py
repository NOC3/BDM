from imgUtils import *
from imgStruct import * 

imgDict = initStruct("./images")



for thisImg in imgDict:

    subImgArray = splitImg(thisImg["obj"], 8)

    colormoment = []
    elbp = []
    hog = []

    for subThisImg in subImgArray:
        avgColor = meanCM(subThisImg)
        stDv = standardDeviationCM(subThisImg, avgColor) 
        skewness= skewnessCM(subThisImg, avgColor)

        colormoment.append({"avg": avgColor,"stdv": stDv,"ske": skewness})

        elbp.append(ELBP(subThisImg))

        
    hog.append(HOG(thisImg["obj"]))

    tmpDict={
        "colormoment": colormoment,
        "elbp": elbp,
        "hog": hog
    }

    thisImg["features"]=tmpDict

print(imgDict)
