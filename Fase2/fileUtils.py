import json


def printToFile(idFile, struct, model, k): #k = numero di feature
    
    features = []
    i = 0
    while i<k: 
        tmpFeat = []
        
        for imgID in struct:
            tmp = { "nome": imgID, "w" : struct[imgID][i]}
            tmpFeat.append(tmp)

        i+=1

        def sortFunc(e):
            return e['w']
        tmpFeat.sort(key=sortFunc, reverse=True) 

        features.append(tmpFeat)

    with open("./outputs/"+idFile, "w")as f:
        for feat in features:
            f.write(str(feat))
            f.write("\n")
        f.close()

def featureMatrixToJSON(struct, idFile):
    dict = {}
    
    for img in struct:
        dict[img["path"]] = img["latentSemantics"].tolist()
    
    jsonmatrix = json.dumps(dict)
    with open("./outputs/"+idFile, "w")as f:
            f.write(jsonmatrix)
            f.close()

def readFeatureMatrix(idFile):
    with open("./outputs/"+idFile, "r")as f:
            jsonString = f.read()
            f.close()
    
    jsonObj = json.loads(jsonString)
    return jsonObj


def featureMatrixToJSON(struct, idFile):
    dict = {}
    
    for img in struct:
        dict[img["path"]] = img["latentSemantics"].tolist()
    
    jsonmatrix = json.dumps(dict)
    with open("./outputs/"+idFile, "w")as f:
            f.write(jsonmatrix)
            f.close()

def printFeatureDict(struct, idFile):
    dict = {}
    
    for key in struct:
        dict[key] = struct[key]#.tolist()
    jsondict = json.dumps(dict)
    
    with open("./outputs/"+idFile, "w")as f:
            f.write(jsondict)
            f.close()
    