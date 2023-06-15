from torch.utils.data import Dataset
#import json
#print(Dataset)
class gpt2train(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = open(path)
        self.Y=[]

        self.X = []
        
        for i in self.data:
            self.Y.append(i.strip())
    
        #for idx,i in enumerate(self.Y):
         #       self.X[idx] = "<startofstring> "+i+" <tag>: "+self.X[idx+1]+" <endofstring>"
        for i in range(0,len(self.Y),2):
            #print(self.Y[i],self.Y[i+1],i,len(self.Y))
            self.X.append( "<startofstring> "+self.Y[i]+" <tag>: "+(self.Y[i+1]).split(':')[1]+" <endofstring>")

        #self.X = self.X[:500]
        
        #print(self.X)

        self.X_encoded = tokenizer(self.X,max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])