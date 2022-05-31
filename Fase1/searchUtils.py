from operator import mod
from statistics import mean
from imgUtils import *
from imgStruct import * 
import math

#metodo che calcola i migliori k sui 3 metodi, se:
# - ci sono almeno k elementi ripetuti si usa il sistema di voto
# [NON APPLICATO] - ci sono <k elementi ripetuti si aggregano i valori delle distanze -> si ordinano per voto
def searchAll(struct, ID, k):
    model = ["moments","ELBP","HOG"]
    totRes = []
    for m in model:
        tmp = search(struct, ID, m, k)
        totRes.append(tmp)
    
    bestKVote, resVotes = vote(totRes, k) #k serve per stimare il voto
    bestKDist, resDist = cmpDistance(totRes, k)


#    for e in totRes:
#        print(e)
#    print("\n")
#    print(resVotes)
#    print(resDist)
#    print("\n")
#    print(bestKVote)
#    print(bestKDist)

    def findID(array, id):
        res = None
        for e in array:
            if(e["id"]== id):
                res = e["distance"]
                break
        return res

    resDict = []
    for e in bestKVote:
        id = e
        voto = resVotes[e]
        cmV = findID(totRes[0], id)
        elbpV = findID(totRes[1], id)
        hogV = findID(totRes[2], id)
        values = {
            "CM": cmV,
            "ELBP": elbpV,
            "HOG": hogV
        }
        res = {
            "id" : id,
            "voto" : voto,
            "valori" : values
        }
        resDict.append(res)

    return bestKVote, resDict

def cmpDistance(totRes, k):
    voteDict = {}
    
    for i in range(len(totRes)):
        for j in range(len(totRes[i])):
            elem = totRes[i][j]
            elem_id = elem["id"]
            vote = elem["distance"]
            
            if elem_id in voteDict.keys():
                voteDict[elem_id] += vote
            else:
                voteDict[elem_id] = vote
    

    sortedVoteDict = (sorted(voteDict, key=voteDict.get))
    
    return sortedVoteDict[0:k], voteDict



def vote(totRes,  k):
    voteDict = {}

    for i in range(len(totRes)):
        for j in range(len(totRes[i])):
            elem = totRes[i][j]
            elem_id = elem["id"]
            vote = (k-j)/k
            
            if elem_id in voteDict.keys():
                voteDict[elem_id] += vote
            else:
                voteDict[elem_id] = vote
    

    sortedVoteDict = (sorted(voteDict, key=voteDict.get))
    sortedVoteDict.reverse()

    return sortedVoteDict[0:k], voteDict

#funzione che restituisce l'ID e il matching score delle k immagini più simili a struct[ID]
def search(struct, ID, model, k):
    actImg = struct[ID]

    resVect = []

    if model == "moments":
        print("Compute MOMENT")
        resVect = momentsDistance(struct, actImg)
    elif model == "ELBP":
        print("Compute ELBP")
        resVect = ELBPDistance(struct, actImg)
    elif model == "HOG":
        print("Compute HOG")
        resVect = HOGDistance(struct, actImg)

    def sortFunc(e):
        return e['distance']
    resVect.sort(key=sortFunc)    
    return resVect[0:k]


def momentsDistance(imgStruct, actImg):
    distanceVect = []
    actId =  actImg["index"]
    for img in imgStruct:
        id = img["index"]
        if id != actId:
            distance = computeMomentsDistance(actImg, img)
            tmp = {"id":id, "distance":distance}
            distanceVect.append(tmp)
        else:
            print("Controllo la stessa immagine: skip")
    return distanceVect

def computeMomentsDistance(actImg, img):
    meanDiff = []
    sdDiff = []
    skwDiff = []

    mImg = img["features"]["moments"]["mean"]
    mActImg = actImg["features"]["moments"]["mean"]
    actMlen =len(mActImg)

    sdImg = img["features"]["moments"]["sd"]
    sdActImg = actImg["features"]["moments"]["sd"]
    actSDlen = len(sdActImg)

    skwImg = img["features"]["moments"]["skw"]
    skwActImg = actImg["features"]["moments"]["skw"]
    actSKWlen = len(skwActImg)


    if len(mImg) == actMlen:
        i = 0
        while i < actMlen:
            tmpDiff = abs(mImg[i] - mActImg[i])
            meanDiff.append(tmpDiff)
            i+=1
        resMean = mean(meanDiff)


    if len(sdImg) == actSDlen:
        i = 0
        while i < actSDlen:
            tmpDiff = abs(sdImg[i] - sdActImg[i])
            sdDiff.append(tmpDiff)
            i+=1
        resSD = mean(sdDiff)


    if len(skwImg) == actSKWlen:
        i = 0
        while i < actMlen:
            tmpDiff = abs(skwImg[i] - skwActImg[i])
            skwDiff.append(tmpDiff)
            i+=1
        resSKW = mean(skwDiff)

    #print("IMG: "+str(img["index"]))
    #print("\tDIFF MEDIA")
    #print("\t"+str(resMean))
    #print("\tDIFF SD")
    #print("\t"+str(resSD))
    #print("\tDIFF SKW")
    #print("\t"+str(resSKW))
        #L1 Minkowski
    resDistance = resMean+resMean+resSKW
    #print("\t\tmul: "+str(resDistance))
    return resDistance



def ELBPDistance(imgStruct, actImg):
    distanceVect = []
    actId =  actImg["index"]
    for img in imgStruct:
        id = img["index"]
        if id != actId:
            distance = computeELBPDistance(actImg, img)
            tmp = {"id":id, "distance":distance}
            distanceVect.append(tmp)
        else:
            print("Controllo la stessa immagine: skip")

    return distanceVect

def computeELBPDistance(actImg, img):
    

    matrixImg = img["features"]["ELBP"]
    matrixActImg = actImg["features"]["ELBP"]   
    lenMatrixActImg = len(matrixActImg)

    res = 0
    summatory = 0
    totObjs = 0
    if(lenMatrixActImg==len(matrixImg)):
        i = 0
        while i<lenMatrixActImg:
            j=0
            while j< len(matrixActImg[i]):
                summatory+= KLDistance(matrixActImg[i][j],matrixImg[i][j])
                totObjs+=1
                j+=1
            i+=1
        res = summatory/totObjs

    else:
        print("Error: different length")
    return  res 



#KLDistance bit to bit
def KLDistance(vecA, vecB):
    sum = 0
    i = 0
    lenA = len(vecA)
    if( lenA ==len(vecB)):

        while i<lenA:
            a = vecA[i]/8
            if a == 0:
                a = 0.0000001
            b = vecB[i]/8 #if == 0 valore fittizio 0.000000000000001
            if b == 0:
                b = 0.0000001

            log = math.log((a/b),10)
            sum+= a*log
            i+=1

    return sum




def HOGDistance(imgStruct, actImg):
    distanceVect = []
    actId =  actImg["index"]
    for img in imgStruct:
        id = img["index"]
        if id != actId:
            distance = computeEuclideanDistance(actImg, img)
            tmp = {"id":id, "distance":distance}
            distanceVect.append(tmp)
        else:
            print("Controllo la stessa immagine: skip")
    return distanceVect


def computeEuclideanDistance(actImg, img):    
    aImg = img["features"]["HOG"]
    aActImg = actImg["features"]["HOG"]
    aImgLen =len(aImg)
    aActImgLen = len(aActImg)

    if aImgLen==aImgLen:
        i=0
        summatory = 0
        while i<aImgLen:
            tmp = abs(aImg[i]-aActImg[i])
            summatory += pow(tmp, 2)
            i+=1
        ret = pow(summatory,1/2)
    else:
        print("Error: different length")

    return ret

