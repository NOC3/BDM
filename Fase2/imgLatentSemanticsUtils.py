from operator import mod
from imgStruct import *
from sklearn.decomposition import TruncatedSVD


#l_semantics(modello_feature, Xutente_tipo, k, tecnicheRD, tipo-peso/soggetto-peso?)
def l_semantics(datas, model, type, k, rd, return_type):

    struct = type_filter(datas,type)
   
    #task1
    #estrarre secondo il modello i dati da datas
    structComputedFeatures = computeStruct(struct, model)
    
    
    #estrazione semantiche latenti con algo di riduzione di dimensionalità
    latentSemanticsList = computeLatentSemantics(structComputedFeatures, rd, k, model)


    return latentSemanticsList



def computeStruct(imgStruct, model):

    match model:
        case "moments":
            return computeMomentsStruct(imgStruct)
        case "ELBP":
            return computeELBPStruct(imgStruct)
        case "HOG":
            return computeHOGStruct(imgStruct)

def computeLatentSemantics(datas, rd, k, model):
    match rd:
        case "PCA":
            tmp = PCA()
        case "SVD":
            tmp = SVD(datas, k, model)
        case "LDA":
            tmp = LDA()
    
    return tmp

def PCA():

    return 0

#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
#usiamo ARPACK poichè usa un eigensolver, dalla teoria eigen vector
def SVD(training_datas, k, model):

    datas = None
    match model:
        case "moments":
            datas = structToMatrixMoments(training_datas)
        case "ELBP":
            i =0
        case "HOG":
            i =0

    SVD = TruncatedSVD(n_components=k,algorithm='arpack')
    fit = SVD.fit_transform(datas)   

    newStruct = addToStruct(fit, training_datas, model)

    return fit, newStruct

def LDA():
    return 0







def addToStruct(fit, struct, model):
    lenFit = len(fit)
    if lenFit !=len(struct):
        return None
    i = 0
    while i<lenFit:
        tmp = {}
        tmp=fit[i]
        struct[i]["latentSemantics"]=tmp
        i+=1


    return struct


#fare anche per HOG e ELBP
def structToMatrixMoments(datas):
    d = []
    
    for img in datas:
        tmp = img["features"]["moments"]["mean"]+ img["features"]["moments"]["sd"]+img["features"]["moments"]["skw"]
        d.append(tmp)
        
    return d

def structToMatrixELBP(datas):
    d = []
    
    for img in datas:
        tmp = img["features"]["ELBP"]
        d.append(tmp)
        
    return d

def structToMatrixHog(datas):
    d = []
    
    for img in datas:
        tmp = img["features"]["HOG"]
        d.append(tmp)
        
    return d




#CLUSTERING
#ritorna una matrice |criterio| x |sem_latenti| (|k|)
def clustering(struct, criterio):
    set = {}
    c = None
    match criterio:
        case "sp":
            c = "Y"
            
        case "tp":
            c = "X"
    
    for img in struct:
        subID = img[c]

        if subID in set.keys(): 
            old = set[subID]
            old.append(img["latentSemantics"])
            set[subID] = old
        else:
            set[subID] = [img["latentSemantics"]]
        
    virtualObjDict ={}
    for idKey in set:
        id = set[idKey]
        i = 0
        virtualFeatures = []
        while i< len(id[0]):#cicla sulle feature
            meanFeature = 0
            j = 0
            sum = 0
            while j < len(id): #cicla su tutti gli oggetti
                sum += id[j][i]
                j+=1

            meanFeature = sum/j
            virtualFeatures.append(meanFeature)
            i+=1
        virtualObjDict[idKey] = virtualFeatures
    
    return virtualObjDict


#ritorniamo la media per ogni feature, l'idea è di creare un oggetto virtuale medio che 
#riassuma tutti i valori per ogni feature, il senso è che all'aggiunta di un nuovo oggetto 
#probabilmente sarà diverso da ogni singolo oggetto ma simile alla media degli oggetti
#Questa cosa è sostenuta dalla quantità di immagini diverse -> immagino che la nuova immagine 
#con alta probabilità sia contenuta nello spazio tra le immagini più dissimi tra loro 