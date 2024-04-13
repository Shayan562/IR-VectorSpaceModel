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
import pickle
from os import path,remove
from math import log2
from heapq import heappop,heappush
from glob import glob

class VectorSpaceModel:
    def __init__(self, ALPHA=0.01, K=10):
        self.docArray=[]
        self.termDocVector={}
        self.docTermVectors={}
        self.termArray=[]
        self.queryVector=[]
        self.championList={}
        self.ALPHA=ALPHA
        self.K=K #champion list size


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
                # with open('./IndexDB/ChampionList.pickle','rb') as f:
                #     self.championList=pickle.load(f)
            except:
                # print("Creating partial indexes")
                self.docTermVectors={}
                self.termArray=[]
                self.createDocVectors()
                self.normalizeDocs()
                self.saveIndex()
        return flag
    
    def computeScore(self):
        numDocs=len(self.docArray)
        '''computing idf then multiplying with tf for score'''
        for key in self.termDocVector.keys():#each term
            df=0
            arr=self.termDocVector[key]
            for i in arr:#get df for a given term
                if(i != 0):#if non-zero tf
                    df+=1
                    
            if(len(arr)<numDocs):#if the array for term is small(add zeros for docs)
                diff=numDocs-len(arr)
                arr.extend([0 for i in range(diff)])
            
            idf=log2(numDocs/df) #getting idf for that term log base 2(n/df)
            for i in range(numDocs):#multiplying idf
                arr[i]*=idf
            self.termDocVector[key]=arr


    def createDocVectors(self):
        '''Create the doc vectors using the term vectors'''
        numDocs=len(self.docArray)
        numTerms=len(self.termDocVector)
        for i in self.docArray:#init vector for each doc
            self.docTermVectors[i]=[0 for i in range(numTerms)]

        index=0
        for key in self.termDocVector.keys():#for each term
            self.termArray.append(key)#add to array for maintaining the right index
            arr=self.termDocVector[key]#getting the tf-idf for that term(all docs)
            for i in range(numDocs):#adding the value to each doc at appropriate index
                self.docTermVectors[self.docArray[i]][index]=arr[i]
            index+=1#index for the next term


    def normalizeDocs(self):
        '''L2 Norm'''
        for doc in self.docTermVectors.keys():#for each doc vector
            arr=self.docTermVectors[doc]#get the vector
            weight=0

            for i in arr:#get the weight
                weight+=i**2

            weight=weight**0.5
            for i in range(len(arr)):#divide each value by the weight
                arr[i]=arr[i]/weight

            self.docTermVectors[doc]=arr


    def createChampionList(self): 
        ''' Creating champion lists of max size K using max heap.
            Term-Doc Matrix is used for this'''
        numDocs=len(self.docArray) 
        for term in self.termDocVector.keys():#for each term
            docs=self.termDocVector[term]#get the docs
            heapSort=[]#using max heap
            count=0
            for i in range(numDocs):
                if(docs[i]!=0):#for non-zero values only
                    '''Python only has min heap built in so mulitplying with -1 to use it like max heap'''
                    heappush(heapSort,(docs[i]*-1,self.docArray[i]))
                    count+=1

            count=(min(len(heapSort),self.K))#in case list size < K
            arr=[]
            for i in range(count):
                arr.append(heappop(heapSort)[1])
            self.championList[term]=arr


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
        using nonZeroIndex for calculating magnitude later(l2 normalizaiton). Saves time/iterations + divide by zero error
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

        docs=set()
        for term in queryTerms:
            if(term in self.championList):
                arr=self.championList[term]
                for d in arr:
                    docs.add(d)

        self.createQueryVector(queryTerms) 

        heapResult=[]
        for doc in docs:
            score=self.cosineScore(doc)
            heappush(heapResult,(score*-1,doc))


        result=[]
        for i in range(len(heapResult)):#run for complete heap or when score<alpha
            doc=heappop(heapResult)
            if(doc[0]*-1<self.ALPHA):
                break
            result.append(doc[1])

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
