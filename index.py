class VectorSpaceModel:
    def __init__(self):
        self.docArray=[]
        self.termDocVector={}
        self.idf={}

    def insert(self, doc, word, tf):
        index=-1#to add the tf for a given doc
        if(doc not in self.docArray):#add doc to list(to maintain correct order)
            self.docArray.append(word)

        index=self.docArray.index(doc)
        #the word already exists in the index
        if(word in self.termDocVector):
            arr=self.termDocVector[word]
            if(len(arr)<=index):#the words list is not big enough
                diff=len(self.docArray)-len(arr)
                arr.extend([0 for i in range(diff)])#extend the word arr(with 0)

            arr[index]+=tf#add tf to correct index
            self.termDocVector[word]=arr#update index
        
        else:#the word does not exist in the index
            arr=[0 for i in range(len(self.docArray))]#make a list long enough for the docs
            arr[index]+=tf
            self.termDocVector[word]=arr
