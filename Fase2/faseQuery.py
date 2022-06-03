from imgLatentSemanticsUtils import *

#prende in input la matrice db_obj-features_latenti e le features_latenti query
#restituisce i noimi delle n imamgini più simili alla query
#NOTA: semanticQuery è un array di features
def query(dictmatrix, semanticQuery, n):
    similarityMatrix = computeSimilarity(dictmatrix, semanticQuery)
    similarityMatrix.sort( reverse=True)

    return similarityMatrix[0:n]





def dictToMatrix(dict):
    return None

def computeSimilarity(dbDictMatrix, semanticQuery):

    keys = dbDictMatrix.keys()
    resDict = {}
    lenSemanticQuery = len(semanticQuery)
    for k in keys:
        row = dbDictMatrix[k]
        #moltiplichiamo row[i] x query[i]
        if(len(row) != lenSemanticQuery):
            print(len(row))
            print(lenSemanticQuery)
            print("ERRORE")
            exit()
        else:
            i=0
            rowProduct = 0
            while i<len(row):
                rowProduct +=row[i]*semanticQuery[i]
                i+=1
        resDict[k] = rowProduct 
    return resDict


def relevantN(similarityDict, n, toPrint=False):
    
    
    dict_sorted = sorted(similarityDict.items(), key= lambda x:x[1],reverse=True)
    if toPrint:
        print("LISTA SIMILI:")
        for d in dict_sorted:
            print("\t- "+str(d[0])+"\t\t["+str(d[1])+"]")
    
    return dict_sorted[0:n]
    

def l_semantics_query(structQuery, lsem_model, model):
    print("\tComputing query...")
    #task1
    #estrarre secondo il modello i dati da datas
    structComputedFeatures = computeStruct(structQuery, model)
    
    #estrazione semantiche latenti con algo di riduzione di dimensionalità
    latentSemanticsList = computeLatentSemanticsQuery(structComputedFeatures, lsem_model, model)

    print("\t\t[done]")

    return latentSemanticsList

def computeLatentSemanticsQuery(training_datas, lsem_model, model):
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

    fitted = lsem_model.transform(datas)   
    newStruct = addToStruct(fitted, training_datas, model)


    return fitted, newStruct, lsem_model


#funzione che prende in input un dizionario e restituisce il dizionario suddiviso per tipo/soggetto con i centroidi
def db_xy_subd(struct, model, type):
    #estrazione e suddivisione in un dizionario keys = tipi con per ogni valore una matrice con gli objs del tipo e le loro features
    dict = criteria_subd_xy(struct, type)
    #per ogni tipo del dizionario troviamo un "centroide"/"mediano" e creiamo una matrice tipo-features con tipo = centroide
    dict_xy = {}
    print("\tComputing centroide...")
    for t in dict:
        c = compute_virtual_obj(dict[t])
        dict_xy[t] = c
        print("\t\t \t"+t+":[done]")
        
    print("\t\t[done]")
    
    return dict_xy


def criteria_subd_xy(struct, type):
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
        dict[t] = set_to_matrix(type_set)
    print("\t\t[done]")

    return dict

def set_to_matrix(type_set):
    d = []
    
    for img in type_set:
        tmp = img["latentSemantics"]
        d.append(tmp)
    return  d
