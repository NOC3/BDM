from imgStruct import *
from imgLatentSemanticsUtils import *
from fileUtils import *


all_models = ["moments","ELBP","HOG"]
all_type = ["cc","con", "emboss", "jitter", "neg", "noise01", "noise02", "original", "poster", "rot", "smooth","stipple"]
all_rd =[ "PCA", "SVD", "LDA"]
all_return_type = ["sp", "tp"]
all_matrix_type = ["tt", "ss"]

model = all_models[0]
type = all_type[0]
k = 2
rd = all_rd[2]
return_type = all_return_type[1]
matrix_type = all_matrix_type[1]

struct = initStruct("./all_tmp")
struct = computeStruct(struct, model)

similarity_matrix, key_dict = matrix_similarity(struct, model, k, matrix_type)
res = computeLatentSemanticsMatrix(similarity_matrix, rd, k)
print(res)

final_dict = {}
i = 0
while i<len(key_dict):
    key = key_dict[i]
    final_dict[key] = res[i]
    i+=1

print(final_dict)
file_name = "MS_"+model+"_"+str(k)+"_"+rd+"_"+return_type+".txt"
printToFile(file_name,final_dict, model, k)