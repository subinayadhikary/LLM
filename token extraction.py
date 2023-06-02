import json
import time as t
import pandas as pd
f=open("ssss.json")
da=json.load(f)
l=[]
for i in da:
    for j in da[i]:
        for k in da[i][j]:
            if(k[0].upper()!=k[0].lower()):#to remove texts that are just numbers or special characters
                l.append(k)
emp=pd.DataFrame(data=l)
#print(emp)

s1=set((",".join(set(emp[2]))).split(","))
#print(s1)
dict={}
p=["homicide_murder","life_imprisonment","riot","physical_assault","evidence_inconsistency","witness_testimony","expert_witness_testimony"]
s=[i for i in s1 if i not in p] #list of all other tags
#print(s)
for i in p:
    dict[i]=list(emp[emp[2].str.contains(i)][0])
'''
for i in s: #loop to generate others tag from tags unused
    dict["others"]=list(emp[emp[2].str.contains(i)][0])
'''
#print(dict)

import openai
openai.api_key = "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"

def get_completion(prompt, model="gpt-3.5-turbo"): 
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


prompt=""
g=open("promptfile.txt","w+")
for i in dict:
    r=min(5,len(dict[i]))
    j=0
    c=0
    while c<r:
        if dict[i][j] not in prompt:
            c=c+1
            m=emp[emp[0]==dict[i][j]].index.values
            for k in m:
                emp=emp.drop(k)
            prompt+="Statement:"+dict[i][j]+". \n"
            g.write("Statement:"+dict[i][j]+". \n")
            prompt+="Tag:"+i+"\n"
            g.write("Tag:"+i+"\n")
        j=j+1

h=open("ssss2.txt","r+")
for i in (h.read()).split(". ")[:10]:
    prompt+="Statement:"+i+". \n"
    g.write("Statement:"+i+". \n")
    prompt+="Tag:"+"Others"+"\n"
    g.write("Tag:"+"Others"+"\n")
h.close
g.close()

#print("prompt: ",prompt)
#print(emp)
dh={}
dh["Statement"]=[]
dh["Assigned tag"]=[]
for j in dict.keys():
    dl=list(emp[emp[2]==j].index.values)
    c=0
    for i in dl:
        if emp[0][i] not in dh["Statement"]:
            dh["Statement"].append(emp[0][i])
            dh["Assigned tag"].append(emp[2][i])
            c=c+1
        if c==13:
            break
    #print(dh)
'''
for j in s: #to generate statements from others tag in the test set
    c=0
    for i in emp.index.values:
        if j in emp[2][i]:
           dh["Statement"].append(emp[0][i])
           dh["Assigned tag"].append(emp[2][i])
           c+=1
        if c==2:
            break
'''

p.append("others")

hm=pd.DataFrame(data=dh)
print(hm)
#hm.to_csv('h2_test.csv', index=False)

ip=list(hm.index.values)
#print(ip)
#r=1005 # The starting index from which checking should be started
l=[]
p=[]
for i in ip:
    #print(hm["Statement"][i])
    custom_prompt = f"""
    Your task is to choose an appropriate tag from 
    the set of TAGS delimited by \' for the Statement delimited by ```, \
    Your choice of tag SHOULD be closely based on the samples provided in Examples delimited by \"\"\" , \
    Do NOT choose any tag out of the tag set.
    Statements in indirect speech are "witness_testimony"
    Statements giving detailed description of wounds are "expert_witness_testimony"
    Statements implying involvement of a large groups are "riot"
    Your answer should be of the format:
        Tag:<appropriate tag>
    
    TAGS: \'{p}\'
    Statement: ```{hm["Statement"][i]}```
    Examples: \"\"\"{prompt}\"\"\"
    """
    response = get_completion(custom_prompt)
    print("Statement:",hm["Statement"][i])
    print("Generated",response)
    print("Assigned Tag:",hm["Assigned tag"][i])
    l.append(response.split(":")[1])
    t.sleep(20)
    promp =  f"""
    Your task is to extract the most appropriate words 
    from the Statement delimited by ```, \
    which resembel the Tag \'{response.split(":")[1]}\'
    Your answer should be of the format:
        <Tag>: <appropriate words>
        
    Statement: ```{hm["Statement"][i]}```
    """
    response2 = get_completion(promp)
    print(response2)
    p.append(response2.split(":")[1])
    t.sleep(20)
hm["generated response"]=l
hm["set of words"]=p
hm.to_csv('hm1_test.csv', index=False)
