import pickle
from os import path,remove,makedirs
from math import log2
from heapq import heappop,heappush
from glob import glob


''' Steps (for creating)
    1. Create Index (term-doc matrix with tf as values) using dictionary(term)-value(list)
        ->create doc array(list) to know which doc has which index
    2. Compute the df then idf and then mulitply with the tf and store in the index
        ->idf = log2(N/idf)
    3. Create the Documnet Vectors(Doc Term Vectore with tf-idf as values) 
        ->create term array(list) to know which term has which index using dictionary(term)-value(list)
            ->term array also used for query processing(query vector + identify terms present in dataset)
    4. Normalize the Document Vectors
        ->L2 norm (the root of sum of square divides each value/weight in documnet vector)
            ->L2 norm also used for the query vector
    5. Create the Champion List (K=10 and non-zero weights) using max-heap
        ->Python has min heap so mulitply score with -1 to use it as max-heap
'''
class VectorSpaceModel:
    def __init__(self):
        self.docArray=[]
        self.termDocVector={}
        self.docTermVectors={}
        self.termArray=[]
        self.queryVector=[]
        self.championList={}
        self.K=10 #champion list size
        self.ALPHA=0.01

    def saveIndex(self):
        try:
            with open('./IndexDB/TermDocVector.pickle','wb') as f:
                pickle.dump(self.termDocVector,f)

            with open('./IndexDB/DocArray.pickle','wb') as f:
                pickle.dump(self.docArray,f)

            with open('./IndexDB/DocTermVector.pickle','wb') as f:
                pickle.dump(self.docTermVectors,f)

            with open('./IndexDB/TermArray.pickle','wb') as f:
                pickle.dump(self.termArray,f)

            with open('./IndexDB/ChampionList.pickle','wb') as f:
                pickle.dump(self.championList,f)
        except:
            #incase there is an issue, delete the remainaing to avoid errors in the index
            files=glob('./IndexDB/*.pickle')
            for file in files:
                remove(file)
        

    def readIndex(self):
        flag=None

        if path.exists('./IndexDB/TermDocVector.pickle') and path.exists('./IndexDB/DocArray.pickle'):
            try:
                with open('./IndexDB/TermDocVector.pickle','rb') as f:
                    self.termDocVector=pickle.load(f)
                with open('./IndexDB/DocArray.pickle','rb') as f:
                    self.docArray=pickle.load(f) 
                flag=True
            except:
                self.docArray=[]
                self.termDocVector={}
            
        if(flag!=None):
            try:
                with open('./IndexDB/DocTermVector.pickle','rb') as f:
                    self.docTermVectors=pickle.load(f)
                with open('./IndexDB/TermArray.pickle','rb') as f:
                    self.termArray=pickle.load(f)
                with open('./IndexDB/ChampionList.pickle','rb') as f:
                    self.championList=pickle.load(f)
            except:
                print("Creating partial indexes")
                self.docTermVectors={}
                self.termArray=[]
                self.championList={}
                self.createDocVectors()
                self.normalizeDocs()
                self.createChampionList()
                self.saveIndex()



        print(flag)
        return flag
    
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
        # print(self.docArray)
        numDocs=len(self.docArray)
        numTerms=len(self.termDocVector)
        for i in self.docArray:
            self.docTermVectors[i]=[0 for i in range(numTerms)]

        index=0
        for key in self.termDocVector.keys():
            self.termArray.append(key) 
            arr=self.termDocVector[key]
            # print(f'{key}:{arr}')
            for i in range(numDocs):
                self.docTermVectors[self.docArray[i]][index]=arr[i]
            index+=1 
        # print(self.termArray)
        # print(self.docTermVectors)

    def normalizeDocs(self):
        for doc in self.docTermVectors.keys():
            arr=self.docTermVectors[doc]
            # print(self.docTermVectors[doc])
            weight=0
            for i in arr:
                weight+=i**2
            for i in range(len(arr)):
                arr[i]=arr[i]/(weight**0.5)
                # if(i==0):
                    # print(f'i:{i}, doc:{doc}, arr:{arr[i]}')
            self.docTermVectors[doc]=arr
            # print(f'Doc {doc}, Weight {weight}')
            # print(arr)
    
    def createChampionList(self):
        
        numDocs=20
        
        for term in self.termDocVector.keys():
            docs=self.termDocVector[term]#get the docs
            heapSort=[]
            count=0
            for i in range(numDocs):
                if(docs[i]!=0):
                    heappush(heapSort,(docs[i]*-1,self.docArray[i]))
                    count+=1
                if(count==self.K):
                    break

            count=(min(len(heapSort),count))#in case list size < K
            arr=[]
            for i in range(count):
                arr.append(heappop(heapSort)[1])
            self.championList[term]=arr

        # for key in self.championList:
        #     print(key)
        #     print(self.termDocVector[key])
        #     print(self.championList[key])

    def cosineScore(self, docID):
        '''Already Normalized(docVecs and query vector) so simple dot product will suffice'''
        score=0
        doc=self.docTermVectors[docID]
        for i in range(len(doc)):#both vectors will have same length
            score+=doc[i]*self.queryVector[i]
        return score

    def createQueryVector(self, queryTerms:list[str]):#after pre-processing
        self.queryVector=[0 for i in self.termArray] #init query array of appropriate lenth with zeros
        if(len(queryTerms)==0):
            return
        
        '''
        for calculating magnitude later(l2 normalizaiton). Saves time/iterations + divide by zero error
        we can have two same terms(maybe) so account for edge case be using set
        '''
        nonZeroIndex=set()
        #tf values for terms in the query vector
        for term in queryTerms:
            if(term in self.termArray):
                index=self.termArray.index(term)
                nonZeroIndex.add(index)
                self.queryVector[index]+=1


        '''normalize the query vector'''
        weight=0
        for index in nonZeroIndex:
            weight+=self.queryVector[index]**2
        
        weight=weight**0.5
        for index in nonZeroIndex:
            self.queryVector[index]/=weight
        
    def evaluateQuery(self, queryTerms):
        ''' 1. Get query->Pre-processed (incl stemming)
            2. Get docs (via champion lists)
            3. Make Query Vector
            4. Get Cosine-Score(docs,q)
            5. Return result (ranked and < ALPHA)'''
        # print(queryTerms)
        docs=set()
        for term in queryTerms:
            arr=self.championList[term]
            for d in arr:
                docs.add(d)
        # print(docs)
        self.createQueryVector(queryTerms) 
        # print(self.queryVector)
        heapResult=[]
        for doc in docs:
            score=self.cosineScore(doc)
            heappush(heapResult,(score*-1,doc))

        # print(heapResult)
        result=[]
        for i in range(len(heapResult)):#run for complete heap or when score<alpha
            doc=heappop(heapResult)
            if(doc[0]*-1<self.ALPHA):
                break
            result.append(doc[1])
        # print(result)
        return result

        

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
