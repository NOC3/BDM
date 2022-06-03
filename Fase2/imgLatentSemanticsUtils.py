from imgStruct import *

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA as PCAsolver
from sklearn.decomposition import LatentDirichletAllocation as LDAsolver
import numpy as np

#funzione wrapper per calcolare la matrice di similarità in base a tipo-tipo o sogegtto-soggetto
def matrix_similarity(struct, model, k, type):
    #estrazione e suddivisione in un dizionario keys = tipi con per ogni valore una matrice con gli objs del tipo e le loro features
    dict = criteria_subd(struct, type, model)
    #per ogni tipo del dizionario troviamo un "centroide"/"mediano" e creiamo una matrice tipo-features con tipo = centroide
    median_dict = []
    keys_dict = []
    print("\tComputing centroide...")
    for t in dict:
        #c = compute_centroide(dict[t])
        c = compute_virtual_obj(dict[t])
        median_dict.append(c)
        keys_dict.append(t)
        print("\t\t[done]")

    print("\t\t[done]")
    

    #calcoliamo la matrice trasposta
    np_matrix = np.array(median_dict)
    np_keys = np.array(keys_dict)

    np_matrix_transpose = np.transpose(np_matrix)
    #moltiplichiamo le due matrici
    similarity_matrix = np.dot(np_matrix, np_matrix_transpose)

    return (similarity_matrix, keys_dict)

    
def criteria_subd(struct, type, model):
    print("\tComputing criteria...")
    match type:
        case "ss":
            c = "Y"
        case "tt":
            c = "X"
    dict = {}
    set = {}
    for img in struct:
        subID = img[c]

        if subID in set.keys(): 
            old = set[subID]
            old.append(img)
            set[subID] = old
        else:
            set[subID] = [img]
    #cicliamo sul set che contiene per ogni tipo tutti gli oggetti
    #otteniamo un dizionario che per ogni tipo ha una maticizzazione degli oggetti

    for t in set:
        type_set = set[t]
        dict[t] = wrapper_to_matrix(type_set,model)
    print("\t\t[done]")

    return dict

#wrapper per estrarre le matrici dalla struct
def wrapper_to_matrix(struct,model):
    match model:
        case "moments":
            return structToMatrixMoments(struct)
        case "ELBP":
            return  structToMatrixELBP(struct)
        case "HOG":
            return  structToMatrixHog(struct)
        

#prende in input un a matrice X-features e ne trova il rappresentante -> come rappresentante si prende l'ggetto mediano
def compute_centroide(matrix):

    #oggetto mediano:
    #calcoliamo la matrice di distanze
    #per ogni obj calcoliamo la media (=> alla distanza media tra questo e tutti gli altri objs)
    #sort, prendiamo l'obj con distanza media minore
    dist_matrix = [[None for i in range(len(matrix))] for j in range(len(matrix))] 
    i = 0
    while i < len(matrix):
        j = 0
        while j < len(matrix):
            if(dist_matrix[j][i]==None):
                #compute
                dist_matrix[i][j] = L2_distance(matrix[i], matrix[j])
            else:
                dist_matrix[i][j] = dist_matrix[j][i]
            j+=1
        i+=1

    dist_dict = {}
    for row in range(len(dist_matrix)):
        sum = 0
        i = 0
        row_obj = dist_matrix[row]
        len_row = len(row_obj)
        while i < len_row:
            sum += row_obj[i]
            i+=1
        res = sum / i 
        dist_dict[row] = res

    dict_sorted = sorted(dist_dict.items(), key= lambda x:x[1],reverse=True)

    centroide = dict_sorted[0][0]
    return matrix[centroide]


        
def compute_virtual_obj(matrix):

    #oggetto mediano:
    #calcoliamo la matrice di distanze
    #per ogni obj calcoliamo la media (=> alla distanza media tra questo e tutti gli altri objs)
    #sort, prendiamo l'obj con distanza media minore


    virtual_obj = []
    i = 0
    while i<len(matrix[0]):
        j = 0
        sum = 0
        while j<len(matrix):
            sum += matrix[j][i]
            j+=1
        tmp = sum/j
        virtual_obj.append(tmp)
        i+=1




    return virtual_obj



def L2_distance(v1, v2):
    lenV1 = len(v1)
    
    if lenV1!=len(v2):
        print("Error: 85 - imgLatentSemantics")
        return None
    i = 0
    summatory = 0
    while i<lenV1:
        summatory += pow(v1[i]-v2[i],2)
        i+=1

    return pow(summatory, 1/2)


#l_semantics(modello_feature, Xutente_tipo, k, tecnicheRD, tipo-peso/soggetto-peso?)
def l_semantics(datas, model, type, k, rd):
    print("\tComputing criteria...")
    if(type!=None):
        struct = type_filter(datas,type)
    else:
        struct = datas
   
    #task1
    #estrarre secondo il modello i dati da datas
    structComputedFeatures = computeStruct(struct, model)
    
    #estrazione semantiche latenti con algo di riduzione di dimensionalità
    latentSemanticsList = computeLatentSemantics(structComputedFeatures, rd, k, model)

    print("\t\t[done]")

    return latentSemanticsList



def computeStruct(imgStruct, model):
    print("\tComputing struct...")
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
            tmp = PCA(datas, k, model)
        case "SVD":
            tmp = SVD(datas, k, model)
        case "LDA":
            tmp = LDA(datas, k, model)
    return tmp

def computeLatentSemanticsMatrix(matrix, rd, k):

    match rd:
        case "PCA":
            model = PCAsolver(n_components=k, svd_solver='arpack')
        case "SVD":
            model = TruncatedSVD(n_components=k,algorithm='arpack')
        case "LDA":
            model = LDAsolver(n_components=k)
    


    fit = model.fit_transform(matrix) 

    return fit


#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
#usiamo ARPACK poichè usa un eigensolver, dalla teoria eigen vector
def SVD(training_datas, k, model):

    datas = None
    match model:
        case "moments":
            datas = structToMatrixMoments(training_datas)
        case "ELBP":
            datas = structToMatrixELBP(training_datas)
        case "HOG":
            datas = structToMatrixHog(training_datas)
        case _:
            datas = training_datas

    SVD = TruncatedSVD(n_components=k,algorithm='arpack')
    
    SVD.fit(datas)
   
    fitted = SVD.transform(datas)   
    newStruct = addToStruct(fitted, training_datas, model)


    return fitted, newStruct, SVD

#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def PCA(training_datas, k, model):
    datas = None
    match model:
        case "moments":
            datas = structToMatrixMoments(training_datas)
        case "ELBP":
            datas = structToMatrixELBP(training_datas)
        case "HOG":
            datas = structToMatrixHog(training_datas)
        case _:
            datas = training_datas

    pca = PCAsolver(n_components=k, svd_solver='arpack')
    

    pca.fit(datas)

    fitted = pca.transform(datas)   

    newStruct = addToStruct(fitted, training_datas, model)


    return fitted, newStruct, pca

def LDA(training_datas, k, model):
    datas = None
    match model:
        case "moments":
            datas = structToMatrixMoments(training_datas)
        case "ELBP":
            datas = structToMatrixELBP(training_datas)
        case "HOG":
            datas = structToMatrixHog(training_datas)
        case _:
            datas = training_datas



    lda = LDAsolver(n_components=k)
    
    lda.fit(datas)
    fitted = lda.transform(datas)   

    newStruct = addToStruct(fitted, training_datas, model)


    return fitted, newStruct, lda






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
        elbp = img["features"]["ELBP"]    
        tmpMag = []
        tmpMin = []
        for e in elbp:
            for ee in e:
                tmpMag.append(float(ee[0])) 
                tmpMin.append(float(ee[1]))
        d.append(tmpMag)  #lasciamo solo i maggiori, perdiamo informazione sul perimetro    

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