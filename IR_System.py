'''
    1. Process Documents
    2. Generate Index(load afterwards)
    3. Get Query
    4. Convert Query
    5. Fetch Query(Cosine Similarity)
    6. Optimize Results
'''
import preProcessing as pp
import index as vsm
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# nltk.download('punkt')

class IRSystem:
    def __init__(self, ALPHA=0.01, K=10):
        self.index=vsm.VectorSpaceModel(ALPHA=ALPHA, K=K)#init vector space model
        self.ps = PorterStemmer()#init porter stemmer of index(if we cant load it) and query
        #init stopwords for index(if we cant load it) and query
        self.stopWords= pp.getFileContent('Stopword-List.txt') #get a preset stopword list
        self.stopWords=word_tokenize(self.stopWords) #toeknize the stopwords

        indexRead=self.index.readIndex()#try to read saved index
        if(indexRead==None):#if we cant read the index
            files=['1','2','3','7','8','9','11','12','13','14','15','16','17','18','21','22','23','24','25','26']#pre difined set of files
            for i in files:#parse each file and insert into index one file at a time
                content= pp.getFileContent('./ResearchPapers/'+i+'.txt')#read the file
                cleanedWords=pp.tokenizeAndClean(content,self.stopWords) # clean the tokens(numbers/punctuations/special characters/stop words) and extract words-count
                for key in cleanedWords.keys():#for each term
                    word=self.ps.stem(key)#stemming
                    self.index.insert(i,word,cleanedWords[key])#insertion

            ''' First we create the term-doc vector(done above) then we build rest of the vectors
                Following tasks are done in order:
                1. tf-idf scores
                2. Creation of Doc-Vecotrs
                3. Normalize The Doc-Term Vectors
                4. Save the created indexes for future use''' 
            self.index.computeScore()
            self.index.createDocVectors()
            self.index.normalizeDocs()
            self.index.saveIndex()#save the index for future
        
        '''Champion list created each time due to the dynamic allocation of champion list size(k)'''
        self.index.createChampionList()


    def queryPreProcessing(self, query):
        '''Clean The query tokens then stem each word'''
        cleanedWords=pp.tokenizeAndClean(query,self.stopWords)
        words=[]
        for key in cleanedWords.keys():
            word=self.ps.stem(key)
            for i in range(cleanedWords[key]):
                words.append(word)
        return words
    

    def runQuery(self,query):
        '''Process the query then run it in the vectore space model(index) and return result'''
        terms=self.queryPreProcessing(query)
        result=self.index.evaluateQuery(terms)
        return result
    

    def goldenSetTest(self):
        '''Golden Set ---Issues with the given answers(manually checked)'''
        queries=['machine learning', 'local global feature','deep convolutional network','intelligent search','transformer','cancer','feature selection machine learning','information retrieval','natural intelligence','artificial intelligence']
        answers=['15,12,1,3,13,2,4,18,20,16,11','16,17,19,18,20,4,1','12,3,2,11,4,5,15,1','NIL','15,14','NIL','16,17,19,18,20,5,12,1,4,3,2,13,11','16,4,1','5','5,1,3']
        for i in range(10):
            q=self.queryPreProcessing(queries[i])
            result=self.index.evaluateQuery(q)
            print(queries[i])
            print(f'Model:   {answers[i].split(',')}')
            print(f'Current: {result}')