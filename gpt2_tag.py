from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gpt2train import gpt2train
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

def train(chatData, model, optim):

    epochs = 60

    for i in tqdm.tqdm(range(epochs)):
        for X, a in chatData:
            #print(X,a)
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")
        print(infer("He was charged under section 302"))

def infer(inp):
    inp = "<startofstring> "+inp+" <tag>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a, no_repeat_ngram_size=2,early_stopping=True,max_new_tokens=10)
    #output=model.generate(inp,no_repeat_ngram_size=2,early_stopping=True,max_new_tokens=3)
    output = tokenizer.decode(output[0])
    return output

#****** test and train set generation*****
import json
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

prompt=""
g=open("promptfile.txt","w+")
for i in dict:
    r=min(20,len(dict[i]))
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


#**********test and train set generation end***********


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>",
                                "sep_token": "<tag>:"})
#tokenizer.add_tokens(["<tag>:"])

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

# print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
#                          return_tensors="pt"))[0]))

chatData = gpt2train("./promptfile.txt", tokenizer)
chatData =  DataLoader(chatData, batch_size=64)
print(chatData)
model.train()

optim = Adam(model.parameters(), lr=1e-3)

print("training .... ")
train(chatData, model, optim)

print("infer from model : ")
'''
while True:
  inp = input()
  print(infer(inp))
''' 
#*****file creation for testing******

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
        if c==20:
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
for i in ip:
    print("Statement:",hm["Statement"][i])
    s = infer(hm["Statement"][i])
    print(s)
    j=s.index('<endofstring>')
    m=s.index('<tag>:')
    #p=s[i:].split(':')[1]
    response=s[m:j].split(':')[1]
    #print(p)
    print("Generated Tag:",response)
    print("Assigned Tag:",hm["Assigned tag"][i])
    l.append(response)
hm["generated response"]=l
hm.to_excel('gpt2_test.xlsx', index=False)

