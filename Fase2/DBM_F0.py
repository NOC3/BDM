from imgStruct import *
from imgLatentSemanticsUtils import *
from fileUtils import *


all_models = ["moments","ELBP","HOG"]
all_type = ["cc","con", "detail", "emboss", "jitter", "neg", "noise1", "noise2", "original", "poster", "rot", "smooth","stipple"]
all_rd =[ "PCA", "SVD", "LDA"]
all_return_type = ["sp", "tp"]


model = all_models[0]
type = all_type[0]
k = 2
rd = all_rd[1]
return_type = all_return_type[0]



struct = initStruct("./all")

res = l_semantics(struct, model, type, k, rd, return_type)

print(res[1])

res1 = clustering(res[1], return_type)

feat_num = len(res[0][0])
file_name = model+"_"+type+"_"+str(k)+"_"+rd+"_"+return_type+".txt"
printToFile(file_name,res1, model, feat_num)