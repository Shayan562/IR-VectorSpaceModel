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
import preProcessing as cf
import PySimpleGUI as sg
# import gui
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')

if __name__=='__main__':
    #first try to load the index- incomplete
    val=None
    if(val!=None):#if we can read it
        pass
    else:#make new index if we cant read the old one/does not exist
        # files=['1','2','3','7','8','9','11','12','13','14','15','16','17','18','21','22','23','24','25','26']
        files=['1']
        for i in files:#parse each file and insert into index one file at a time
            content= cf.getFileContent('./ResearchPapers/'+i+'.txt')#read the file
            stopWords= cf.getFileContent('Stopword-List.txt') #get a preset stopword list
            stopWords=word_tokenize(stopWords) #toeknize the stopwords
            cleanedWords=cf.tokenizeAndClean(content,stopWords) # clean the tokens(numbers/punctuations/special characters/stop words)
        