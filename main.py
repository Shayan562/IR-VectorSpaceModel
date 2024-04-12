import gui

if __name__=='__main__':
    app=gui.App()
    app.mainloop()

"""




class ResultsWindow(customtkinter.CTkToplevel):
    def __init__(self, query, queryResult):
        super().__init__()
        self.title('Search Results')
        self.customLabels=[]
        self.checkBoxes=[]
        self.row=1
        # self.geometry("{}x{}".format(200, self.winfo_height()))

        self.label = customtkinter.CTkLabel(self, text="Result: "+query)
        self.label.pack(padx=20, pady=20)

        #send query to model and display output down below
        index=1
        for terms in queryResult:
            self.createLabels(str(index)+") Doc "+terms)
            index+=1

    def createLabels(self,text):
        self.customLabels.append(customtkinter.CTkLabel(self, text=text))
        self.customLabels[-1].pack(padx=4,pady=2)
        self.checkBoxes.append(customtkinter.CTkCheckBox(self,text='Relevant',border_width=1,checkbox_width=12,checkbox_height=12))
        self.checkBoxes[-1].pack(padx=0,pady=0)
"""