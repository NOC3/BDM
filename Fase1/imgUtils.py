import matplotlib as mat
import matplotlib.pyplot as pl
import matplotlib.image as img
from skimage.feature import hog


import copy

from os import listdir
from os.path import isfile, join




def showImg(imgObj):
    pl.imshow(imgObj)
    pl.show()

def showMoreImg(imgStruct):
    fig = pl.figure(figsize=(10, 7))
    rows = 1
    columns = len(imgStruct)
    i = 1
    while i-1<len(imgStruct):
        fig.add_subplot(rows, columns, i)
        pl.imshow(imgStruct[i-1]["obj"])
        pl.axis('off')
        pl.title(i-1)
        i+=1
    pl.show()


def splitImg(imObj, dimSubImgs):
    x, y = imObj.shape
    xIter = int(x / dimSubImgs)
    yIter = int(y / dimSubImgs)
    
    objResList = []

    #cicliamo sull'imagine e separiamo delle sottoimmagini di dimesione noSubImg
    for i in  range(yIter):
        oldX = 0
        oldY = 0
        newX = dimSubImgs
        newY = dimSubImgs
        for j in range(xIter):
            objResList.append(imObj[oldX:newX,oldY:newY])
            oldX = newX
            oldY = newY
            newX += dimSubImgs
            newY += dimSubImgs
    
    return objResList

#primo momento - media (si sommano i valori dei colori e si divide per il numero di pixel)
def meanCM(imObj):
    x, y = imObj.shape
    sum = 0
    for i in  range(y):
        for j in range(x):
            sum +=imObj[i,j]

    res = sum / (x*y)
    return res

#secondo momento - devizione standard (deviazione standard del colore dalla media)
def standardDeviationCM(imObj, avg):
    x, y = imObj.shape
    sum = 0
    for i in  range(y):
        for j in range(x):
            sumTmp = imObj[i,j]-avg
            sum += pow(sumTmp,2)

    res = sum / (x*y)
    return pow(res, 1/2)

#terzo momento (come deviazione standard ma con esponente 3)
def skewnessCM(imObj, avg):
    x, y = imObj.shape
    sum = 0
    for i in  range(y):
        for j in range(x):
            sumTmp = imObj[i,j]-avg
            sum += pow(sumTmp,3)

    res = abs(float(sum / (x*y)))
    res1 = pow(res, 1/3)
    return res1

#funzione che prende in input una sottoimmagine
#per ogni pixel analizza gli 8 vicini e calcola quanti hanno un valore maggiore e quanti un valore minore del pixel centrale
#return : un array di dimensione numero di pixel dell'immagine contenente in ogni cella quanti vicini sono maggiori e quanti vicini sono minori
#actual return : matrice 2x#pixel -> matrice[0] > del centrale | matrice[1] < del centrale 

#NOTA: i pixel sul margine, che quindi non hanno 8 vicini si comportano:
# si inserisce il valore fittizio -1 che non andrà poi ad essere computato nel conteggio dei maggiori/minori
# motivazione: questo permette di capire implicitamente (dalla somma di maggiori+minori) se un pizel è interno (somma 8) sul lato (somma 5) o sull'angolo (somma 3)

def ELBP(imObj):
    x, y = imObj.shape
    resVect = []


    for i in  range(y):
        for j in range(x):
            pixelValue = imObj[i,j]
            vicini = computeVicini(i,j)            
            mag = 0
            min = 0
            for v in vicini:                    
                if(v[0]<0 or v[1]<0 or v[0]>=x or v[1]>=y):
                    #serve a verificare che la posizione nella matrice esista veramente
                    continue
                else:
                    vicinoColor = imObj[v[0], v[1]]
                    if(pixelValue>=vicinoColor):
                        mag +=1
                    else:
                        min+=1
            resVect.append((mag,min))
    return resVect



#funzione che dati due indici (e quindi un punto sulla matrice) ti ritorna gli indici dei vicini
def computeVicini(r,c):
    vicini = []
    vicini.append((r-1,c))
    vicini.append((r-1,c+1))
    vicini.append((r,c+1))
    vicini.append((r+1,c+1))
    vicini.append((r+1,c))
    vicini.append((r+1,c-1))
    vicini.append((r,c-1))
    vicini.append((r-1,c-1))
    return vicini


def HOG(imObj):
    h, img = hog(imObj, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True, feature_vector=True)
    return h