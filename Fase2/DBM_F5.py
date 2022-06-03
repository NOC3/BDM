from imgStruct import *
from imgLatentSemanticsUtils import *
from fileUtils import *
from faseQuery import *


all_models = ["moments","ELBP","HOG"]
all_type = ["cc","con", "emboss", "jitter", "neg", "noise01", "noise02", "original", "poster", "rot", "smooth","stipple"]
all_rd =[ "PCA", "SVD", "LDA"]
all_matrix_type = ["tt", "ss"]

model = all_models[0]
type = all_type[0]
k_db = 100
rd = all_rd[0]
matrix_type = all_matrix_type[1]

#numero di risultati rilevanti
n = 5

DB_struct = initStruct("./all_tmp")

#riempiamo il db con le semantiche latenti di tutte le immagini
DB_l_semantics, DB_struct_lsem, lsem_model = l_semantics(DB_struct, model, None, k_db, rd)

'''FINO A QUA  con 4 esecuzioni rimane tutto uguale -> hog leggermente diverso'''


idFileFeatureMatrix = "FM_"+model+"_"+str(k_db)+"_"+rd+"_"+type+".txt"
#scriviamo su file la matice obj-feature
featureMatrixToJSON(DB_struct_lsem, idFileFeatureMatrix)

#calcoliamo le semantiche latenti della query
structQuery = initStruct("./obj_query")
Q_l_semantics, Q_struct_lsem, lsem_model = l_semantics_query(structQuery, lsem_model, model)



#recuperiamo da databasei dati
jsonDict = readFeatureMatrix(idFileFeatureMatrix)

#confronto query-bd_objs
resQuery = computeSimilarity(jsonDict, Q_l_semantics[0])# Q_l_semantics[0] -> [0] perchè viene restituito come matrice di 1 riga

similarImgNames = relevantN(resQuery, n, toPrint=False)


#print delle immagini
#print(similarImgNames)


for s in similarImgNames:
    print(s)

print("\n")
print(resQuery["./all_tmp/image-cc-1-1.png"])


showQueryImg(structQuery[0],similarImgNames)
