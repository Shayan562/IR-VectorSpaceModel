'''
    1. Process Documnets
    2. Generate Index(load afterwards)
    3. Get Query
    4. Convert Query
    5. Fetch Query(Cosine Similarity)
    6. Optimize Results
'''

import nltk
# import indexDataStructures as ds
import preProcessing as pp
import PySimpleGUI as sg
import index as vsm
# import gui
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# nltk.download('punkt')

def queryPreProcessing(query,stopWords,ps):
    stopWords= pp.getFileContent('Stopword-List.txt')
    cleanedWords=pp.tokenizeAndClean(query,stopWords)
    words=[]
    for key in cleanedWords.keys():
        word=ps.stem(key)
        words.append(word)
        # index.insert(i,word,cleanedWords[key])
    return words

if __name__=='__main__':
    #first try to load the index- incomplete
    index=vsm.VectorSpaceModel()
    val=index.readIndex()
    ps = PorterStemmer()
    stopWords= pp.getFileContent('Stopword-List.txt') #get a preset stopword list
    stopWords=word_tokenize(stopWords) #toeknize the stopwords
    if(val==None):#if we cant read it
        # print("Creating INdex")
        # files=['1']
        files=['1','2','3','7','8','9','11','12','13','14','15','16','17','18','21','22','23','24','25','26']
        for i in files:#parse each file and insert into index one file at a time
            content= pp.getFileContent('./ResearchPapers/'+i+'.txt')#read the file
            cleanedWords=pp.tokenizeAndClean(content,stopWords) # clean the tokens(numbers/punctuations/special characters/stop words)
            for key in cleanedWords.keys():
                word=ps.stem(key)
                index.insert(i,word,cleanedWords[key])

                # if(word=='overview'):
                #     print(f"{i},{word}:{cleanedWords[key]}")
                    
        index.computeScore()
        index.createDocVectors()
        index.normalizeDocs()
        index.createChampionList()
        index.saveIndex()#save the craetd index for future
        
    
    # print(len(index.termArray))
    queries=['machine learning', 'local global feature','deep convolutional network','intelligent search','transformer','cancer','feature selection machine learning','information retrieval','natural intelligence','artificial intelligence']
    answers=['15,12,1,3,13,2,4,18,20,16,11','16,17,19,18,20,4,1','12,3,2,11,4,5,15,1','NIL','15,14','NIL','16,17,19,18,20,5,12,1,4,3,2,13,11','16,4,1','5','5,1,3']
    for i in range(10):
        q=queryPreProcessing(queries[i],stopWords,ps)
        result=index.evaluateQuery(q)
        print(queries[i])
        print(f'Model:   {answers[i].split(',')}')
        print(f'Current: {result}')



#"""