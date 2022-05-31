def printToFile(idFile, struct, model, k): #k = numero di feature
    
    features = []
    print(struct)
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
