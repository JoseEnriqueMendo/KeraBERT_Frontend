class ClinicalCasesToParagraph:
    parrafo=""
    def __init__(self, df):
        self.df = df

    def makeSentence(self,response,context,flag):
        sentence=""
        if response=="none":
            sentence="without "+context
            return sentence

        if response=="no":
            sentence="not "+context
            return sentence

        if flag==1:
            return "with "+response
        elif flag==2:
            return "with "+context
        elif flag==3:
            return  "with "+response+" "+context
        elif flag==4:
            return  "with "+context+" "+response
        elif flag==5:
            return  context
        else:
            return "ERROR"

    def makeParagraph(self,row):
        gender=str(row.loc['Género'])
        years=str(row.loc['Edad en años'])

        CME=str(row.loc['Condición médica especial'])
        CME=self.makeSentence(CME,"special medical condition",1)

        glasses=str(row.loc['Uso de ayudas ópticas'])
        glasses=self.makeSentence(glasses,"glasses",1)

        satisfied=str(row.loc['Satisfecho con la corrección óptica actual'])
        satisfied=self.makeSentence(satisfied,"satisfied with the correction",5)

        Intolerance=str(row.loc['Intolerancia a lentes de contacto'])
        Intolerance=self.makeSentence(Intolerance,"intolerance to contact lenses",2)

        Frequent_eye_rub=str(row.loc['Frote ocular frecuente'])
        Frequent_eye_rub=self.makeSentence(Frequent_eye_rub,"frequent eye rub",2)

        CDVA=str(row.loc['Agudeza visual con corrección'])
        CDVA=self.makeSentence(CDVA,"CDVA",4)

        UDVA=str(row.loc['Agudeza visual sin corrección'])
        UDVA=self.makeSentence(UDVA,"UDVA",4)

        K2=str(row.loc['K2'])
        K2=self.makeSentence(K2,"k2 of",4)

        Kmax=str(row.loc['K max'])
        Kmax=self.makeSentence(Kmax,"kmax of",4)

        Thinnest=str(row.loc['Thinnest point pentacam'])
        Thinnest=self.makeSentence(Thinnest+" ","corneal thickness of",4)

        progressive_keratoconus=str(row.loc['Queratocono progresivo'])
        progressive_keratoconus=self.makeSentence(progressive_keratoconus,"progressive keratoconus",2)

        corneal_hydrops=str(row.loc['Hidrops corneal'])
        corneal_hydrops=self.makeSentence(corneal_hydrops,"corneal hydropsis",2)

        corneal_scar=str(row.loc['Cicatriz corneal'])
        corneal_scar=self.makeSentence(corneal_scar,"corneal scar",3)

        cylinder=str(row.loc['Cilindro'])
        cylinder=self.makeSentence(cylinder,"",1)

        eye=str(row.loc['Ojo'])
        if eye=="OD":
            eye="right eye"
        else:
            eye="left eye"

        eye=self.makeSentence(eye,"keratoconus in the",4)

        paragraph= years+" year old "+gender+", "+CME+", "+glasses+", "+satisfied+", "+Intolerance+", "+Frequent_eye_rub+", "+CDVA+", "+UDVA+", "+K2+"D"+", "+Kmax+"D"+", "+Thinnest+" microns, "+progressive_keratoconus+", "+corneal_hydrops+", "+corneal_scar+", "+cylinder+"D cylinder and "+eye

        return paragraph

    def returnParagraph(self):
        self.df=self.df.replace({"Femenino": "female","Masculino": "male","Ninguna": "none","Lentes": "glasses","Antecedente de herpes ocular": "history of ocular herpes","Queratoplastia": "Keratoplasty","No": "no","Cuenta dedos": "count fingers","Sí": "yes","Profunda": "deep", "Anterior": "anterior"})
        for r in self.df.iloc:
            self.parrafo=self.makeParagraph(r)

        return self.parrafo