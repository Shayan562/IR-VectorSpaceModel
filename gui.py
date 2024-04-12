import customtkinter
import IR_System

class App(customtkinter.CTk):
    def __init__(self):

        self.engine=IR_System.IRSystem()
        # self.engine.goldenSetTest()


        super().__init__()

        customtkinter.set_appearance_mode("dark")
        self.title("Vector Space Model")
        self.geometry("400x180")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        # self.search_frame = customtkinter.CTkFrame(self)
        # self.search_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky='ew')

        self.searchBox = customtkinter.CTkTextbox(master=self,height=50,corner_radius=15,wrap='word',activate_scrollbars=True)
        self.searchBox.grid(row=0,padx=20,pady=(20,0),sticky='ew')
        self.searchBox.insert("0.0", "Enter Query",tags=None)
        # self.checkbox_1 = customtkinter.CTkCheckBox(self.checkbox_frame, text="checkbox 1")
        # self.checkbox_1.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        # self.checkbox_2 = customtkinter.CTkCheckBox(self.checkbox_frame, text="checkbox 2")
        # self.checkbox_2.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")

        self.button = customtkinter.CTkButton(self, text="Search", command=self.button_callback)
        self.button.grid(row=1, column=0, padx=10, pady=10,sticky='n')


        self.searchResults = None

    def button_callback(self):
        query=self.searchBox.get('0.0','end').rstrip('\n')
        if(query==' ' or query==''):
            return
        
        self.searchBox.delete('0.0','end')
        result=self.engine.runQuery(query)

        if(len(result)==0):
            result.append('None')

        if self.searchResults is None or not self.searchResults.winfo_exists():
            self.searchResults = ResultsWindow(query,result)  # create window if its None or destroyed
        else:
            self.searchResults.focus()  # if window exists focus it

        # print(query)
        # print("button pressed")
class ResultsWindow(customtkinter.CTkToplevel):
    def __init__(self, query, queryResult):
        super().__init__()
        self.title('Search Results')
        self.customLabels=[]
        self.checkBoxes=[]
        self.row=0
        self.minsize(width=300,height=250)
        # self.geometry("350x600")

        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=1)
        self.rowconfigure(1,weight=60)
        self.result_frame=customtkinter.CTkFrame(self)
        self.result_frame.grid(row=1, column=0, padx=0, pady=(10, 0), sticky='nsew')
        self.result_frame.columnconfigure(0,weight=1)
        self.result_frame.columnconfigure(1,weight=1)

        # self.geometry("{}x{}".format(200, self.winfo_height()))

        self.label = customtkinter.CTkLabel(self, text="Result: "+query)
        # self.label.pack(padx=20, pady=20)
        self.label.grid(row=0,column=0,padx=20, pady=(10,0), sticky='n')

        #send query to model and display output down below
        index=1
        if(queryResult[0]=='None'):
            self.customLabels.append(customtkinter.CTkLabel(self.result_frame, text='None'))
        # self.customLabels[-1].pack(padx=4,pady=2)
            self.customLabels[-1].grid(row=self.row,column=0,padx=200,pady=2)
            return
        for terms in queryResult:
            self.createLabels(str(index)+") Doc "+terms)
            index+=1

        self.precisonButton = customtkinter.CTkButton(self, text="Get Precision", command=self.calculatePrecision)
        self.precisonButton.grid(row=2, column=0, padx=10, pady=10,sticky='n')
        

    def calculatePrecision(self):
        precision=0
        count=0
        for i in range(len(self.checkBoxes)):
            box=self.checkBoxes[i]
            if(box.get()==1):
                count+=1
                precision+=(count/(i+1))
                # print(f'{i}: Count:{count}, Precision{precision}')

        # print(precision/count)
        if(count!=0):
            precision/=count
        print(precision)
        precision="{:.2f}".format(precision)
        self.precisionLabel=customtkinter.CTkLabel(self, text='Average Precision: '+precision)
        self.precisionLabel.grid(row=3,column=0,pady=5)

    def createLabels(self,text):
        # self.search_frame = customtkinter.CTkFrame(self)
        # self.search_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky='ew')

        self.customLabels.append(customtkinter.CTkLabel(self.result_frame, text=text))
        # self.customLabels[-1].pack(padx=4,pady=2)
        self.customLabels[-1].grid(row=self.row,column=0,padx=4,pady=2)
        self.checkBoxes.append(customtkinter.CTkCheckBox(self.result_frame,text='Relevant',border_width=1,checkbox_width=12,checkbox_height=12))
        # self.checkBoxes[-1].pack(padx=0,pady=0)
        self.checkBoxes[-1].grid(row=self.row,column=1,padx=4,pady=2)
        self.row+=1





