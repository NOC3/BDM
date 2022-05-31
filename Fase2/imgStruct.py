from os import listdir
from os.path import isfile, join

from imgUtils import *


def initStruct(importPath):
    #inizializzazione dizionari imgs
    imgPath = importPath
    setImg = []
    onlyfiles = []
    for f in listdir(imgPath):
        onlyfiles.append(join(imgPath, f))

    #costruzione dizionario
    index = 0
    imgDict = []
    for file in onlyfiles:
        xyz =  file.replace(".png", "").split("-")
        tmpImg = img.imread(file)
        tmpDic = {
            "obj" : tmpImg,
            "index": index,
            "path" : file,
            "X": xyz[1],
            "Y": xyz[2],
            "Z": xyz[3],
            "features": {
                "moments":None,
                "ELBP": None,
                "HOG": None
            }
        }
        imgDict.append(tmpDic)
        index +=1

    return imgDict

def type_filter(struct, type):
    ret = []
    for s in struct:
        if s["X"] == type:
            ret.append(s)
    return ret


#funzione che calcola e assegna alla struttura le varie features
def computeFeatures(imgStruct):
    imgStruct = computeMomentsStruct(imgStruct)
    imgStruct = computeELBPStruct(imgStruct)
    imgStruct = computeHOGStruct(imgStruct)
    return imgStruct

#wrapper per inserire nella struttura la feature "moments"
def computeMomentsStruct(imgStruct):
    i = 0
    for img in imgStruct:
        img["features"]["moments"] = computeMoments(img)
        i+=1
    
    return imgStruct


#funzione che restituisce la struttura dei momenti per un immagine
def computeMoments(mainImg):
    tmpMean = []
    tmpSD = []
    tmpSKW = []

    subImgArray = splitImg(mainImg["obj"], 8)


    for img in subImgArray:
        avg = meanCM(img)
        tmpMean.append(avg)
        tmpSD.append(standardDeviationCM(img, avg))
        tmpSKW.append(skewnessCM(img, avg))

    res = {
        "mean": tmpMean,
        "sd": tmpSD,
        "skw": tmpSKW
    }

    return res



def computeELBPStruct(imgStruct):
    for img in imgStruct:
        img["features"]["ELBP"] = computeELBP(img)
    return imgStruct

def computeELBP(mainImg):

    subImgArray = splitImg(mainImg["obj"], 8)
    
    tmpELBP = []
    for img in subImgArray:
        tmpELBP.append(ELBP(img))

    return tmpELBP


def computeHOGStruct(imgStruct):
    for img in imgStruct:
        img["features"]["HOG"] = computeHOG(img)
    return imgStruct

def computeHOG(mainImg):
    return hog(mainImg['obj'])