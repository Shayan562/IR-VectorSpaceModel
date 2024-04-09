import pickle
from os import path
from math import log2
class VectorSpaceModel:
    def __init__(self):
        self.docArray=[]
        self.termDocVector={}

    def saveIndex(self):
        with open('./TermDocVector.pickle','wb') as f:
            pickle.dump(self.termDocVector,f)

        with open('./DocArray.pickle','wb') as f:
            pickle.dump(self.docArray,f)

    def readIndex(self):
        if path.exists('./TermDocVector.pickle') and path.exists('./DocArray.pickle'):
            try:
                with open('./TermDocVector.pickle','rb') as f:
                    self.termDocVector=pickle.load(f)
                with open('./DocArray.pickle','rb') as f:
                    self.docArray=pickle.load(f) 
                return True
            except:
                self.docArray=[]
                self.termDocVector={}
                return None
        return None
    
    def computeScore(self):
        # self.idf={}
        numDocs=len(self.docArray)
        #computing idf then multiplying with tf for score
        for key in self.termDocVector.keys():#each term
            df=0
            arr=self.termDocVector[key]
            for i in arr:#get df for a given term
                if(i != 0):#if non-zero tf
                    df+=1
            """print(f'DF -> {key}: {df}')
            print(f'Vec-> {key}: {self.termDocVector[key]}')"""
            if(len(arr)<numDocs):#if the array for term is small(add zeros for docs)
                diff=numDocs-len(arr)
                arr.extend([0 for i in range(diff)])
            """print(f'Vec-> {key}: {self.termDocVector[key]}')"""
            #getting idf for that term log base 2(n/df)
            idf=log2(numDocs/df)
            # self.idf[key]=log2(numDocs/df)
            """print(f'IDF-> {key}: {idf}')"""
            for i in range(numDocs):
                arr[i]*=idf
            self.termDocVector[key]=arr
            """print(f'Vec-> {key}: {self.termDocVector[key]}')"""

    def createDocVectors(self):
        self.docTermVectors={}
        self.termArray=[]
        print(self.docArray)
        numDocs=len(self.docArray)
        first =True
        numTerms=len(self.termDocVector)
        for i in self.docArray:
            self.docTermVectors[i]=[0 for i in range(numTerms)]

        index=0
        for key in self.termDocVector.keys():
            self.termArray.append(key) 
            arr=self.termDocVector[key]
            # print(key)
            print(f'{key}:{arr}')
            for i in range(numDocs):
                self.docTermVectors[self.docArray[i]][index]=arr[i]
            index+=1 
        print(self.termArray)
        print(self.docTermVectors)


            
        

    def insert(self, doc, word, tf):
        index=-1#to add the tf for a given doc
        if(doc not in self.docArray):#add doc to list(to maintain correct order)
            self.docArray.append(doc)

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
