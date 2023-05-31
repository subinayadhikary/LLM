import pandas as pd
fd=pd.read_csv("hm1_test.csv")
print(fd)
p=["homicide_murder","life_imprisonment","riot","physical_assault","evidence_inconsistency","witness_testimony","expert_witness_testimony"]
dd={}
dd1={}
for i in p:
    tp=0
    fp=0
    fn=0
    tn=0
    for (m,k) in zip(fd["Assigned tag"],fd["generated response"]):
        for j in (m.strip()).split(","):
            if i==j and i==k.strip():
                tp+=1
            elif i!=j and i==k.strip():
                fp+=1
            elif i==j and i!=k.strip():
                fn+=1
            elif i!=j and i!=k.strip():
                tn+=1
    dd[i]=[tp,fp,fn,tn]

'''
tp,fp,fn,tn=0,0,0,0
for (i,j) in zip(fd["Assigned tag"],fd["generated response"]):
    if i not in p:
        if "others" in j:
            tp+=1
        else:
            fn+=1
    else:
        if "others" in j:
            fp+=1
        else:
            tn+=1
dd["others"]=[tp,fp,fn,tn]
'''

print(dd)
#p.append("others")
dd1["labels"]=p
dd1["Accuracy"]=[(dd[i][0]+dd[i][3])/(dd[i][0]+dd[i][1]+dd[i][2]+dd[i][3]) for i in dd.keys()]
dd1["Misclassification"]=[(dd[i][1]+dd[i][2])/(dd[i][0]+dd[i][1]+dd[i][2]+dd[i][3]) for i in dd.keys()]
dd1["Precision"]=[dd[i][0]/(dd[i][0]+dd[i][1]) for i in dd.keys()]
dd1["Sensitivity"]=[dd[i][0]/(dd[i][0]+dd[i][2]) for i in dd.keys()]
dd1["Specificity"]=[dd[i][3]/(dd[i][3]+dd[i][1]) for i in dd.keys()]
em=pd.DataFrame(data=dd1)
print(em)
em.to_csv('data_eval.csv', index=False)