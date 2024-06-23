from pyswip import Prolog
from shapely.geometry import MultiPolygon
import pandas as pd
import joblib
from util import removeDuplicates, getBasePath
import os

'''
Classe che rappresenta la base di conoscenza in prolog e fornisce i metodi per manipolarla e interrogarla
'''
class KB:
    '''
    Costruttore della classe. Inizializza la base di conoscenza in prolog
    Parametri:
    - areasGdf (GeoDataFrame): un GeoDataFrame contenente le aree di Chicago con il loro perimetro
    - modelPath (String): Il percorso del modello di machine learning addestrato per la previsione della gravità dei crimini
    - week (Int): la settimana in cui si vuole effettuare la previsione
    - day (Int): il giorno della settimana in cui si vuole effettuare la previsione
    - timeslot (int): la fascia oraria in cui si vuole effettuare la previsione
    '''
    def __init__(self, areasGdf, modelPath, week, day, timeslot):
        self.prolog = Prolog()
        self.areasGdf = areasGdf
        self.modelPath = modelPath
        self.week = week
        self.day = day
        self.timeslot = timeslot
        self.initializaKB()
    

    '''
    Metodo che inizializza la base di conoscenza in prolog
    '''
    def initializaKB(self):
        kb = str(os.path.join(getBasePath(), "src", "kb.pl")).replace("\\", "/")
        self.prolog.consult(kb)
        # Definizione dei fatti nearAreas per le aree vicine
        self.defineNearAreas()
        # Definizione dei fatti relativi alla gravità delle aree
        self.defineAreaSeverities()
        # Definizione dei fatti relativi alle dimensioni delle aree
        self.defineAreasSize()


    '''
    Metodo che definisce i fatti nearAreas per le aree vicine, della forma nearAreas(AreaNumber1, AreaNumber2).
    Due aree sono considerate vicine se hanno due punti del loro perimetro in comune
    '''
    def defineNearAreas(self):
        i = 0
        y = 0
        for area1 in self.areasGdf.itertuples():
            perimeter1 = MultiPolygon(area1.Perimeter)
            y = 0
            list = []
            for area2 in self.areasGdf.itertuples():
                if i != y:
                    perimeter2 = MultiPolygon(area2.Perimeter)
                    if perimeter1.touches(perimeter2):
                        list.append(area2.AreaNumber)
                y += 1
            self.prolog.assertz(f"area({area1.AreaNumber})")
            self.prolog.assertz(f"nearAreas(area({area1.AreaNumber}), {list})")
            i += 1
    

    '''
    Metodo che definisce i fatti areaSeverity per le aree di Chicago, della forma areaSeverity(AreaNumber, Severity).
    La gravità dei crimini è prevista utilizzando il modello di machine learning addestrato
    '''
    def defineAreaSeverities(self):
        #modelFile = f"models/{self.modelName}_model.pkl"
        loadedModel = joblib.load(self.modelPath)

        for area in self.areasGdf.itertuples():
            data = {
                'Week': self.week,
                'Day': self.day,
                'Time Slot': self.timeslot,
                'Community Area': area.AreaNumber
            }
            prediction = loadedModel.predict(pd.DataFrame(data, index=[0]))
            self.prolog.assertz(f"severity(area({area.AreaNumber}), {int(prediction[0])})")

    
    '''
    Metodo che definisce i fatti size per le aree di Chicago, della forma size(area(AreaNumber), AreaSize).
    Rappresentano le aree delle zone di Chicago
    '''
    def defineAreasSize(self):
        for area in self.areasGdf.itertuples():
            self.prolog.assertz(f"size(area({area.AreaNumber}), {area.AreaSize})")


    '''
    Metodo che imposta l'area come pattugliata. Aggiunge il fatto patrolArea per l'area specificata, della forma patrolArea(AreaNumber)
    Parametri:
    - areaNum (Int): il numero dell'area da impostare come pattugliata
    '''
    def setAreaPatrol(self, areaNum, patrol):
        # controllo se esiste già un fatto patrolArea per l'area specificata
        res = list(self.prolog.query(f"patrolArea(area({areaNum}), _)"))
        if res:
            self.prolog.retract(f"patrolArea(area({areaNum}), _)")
        self.prolog.assertz(f"patrolArea(area({areaNum}), {'true' if patrol else 'false'})")

    

    '''
    Metodo che rimuove l'area dalla lista delle aree pattugliate. Rimuove il fatto patrolArea per l'area specificata
    Parametri:
    - areaNum (Int): il numero dell'area da rimuovere dalla lista delle aree pattugliate
    '''
    def removeAreaPatrol(self, areaNum):
        if list(self.prolog.query(f"patrolArea(area({areaNum}), _)")):
            self.prolog.retract(f"patrolArea(area({areaNum}), _)")


    '''
    Metodo che verifica se l'area specificata è sicura
    Parametri:
    - areaNum (Int): il numero dell'area da verificare
    '''
    def isAreaSafe(self, areaNum):
        res = list(self.prolog.query(f"isSafe(area({areaNum}))"))
        return res
    

    '''
    Metodo che restituisce la lista di tutte le aree di Chicago
    '''
    def getAreasList(self):
        areasList = []
        res = list(self.prolog.query("area(X)"))
        for area in res:
            areasList.append(area['X'])
        return areasList
    

    '''
    Metodo che restituisce la lista delle aree di Chicago a una distanza specificata dall'area specificata
    Parametri:
    - areaNum (Int): il numero dell'area di partenza
    - distance (Int): la distanza dall'area di partenza
    '''
    def getAreasByDistance(self, areaNum, distance):
        areas = list(self.prolog.query(f"distance(area({areaNum}), area(X), {distance})"))
        areasList = []
        for area in areas:
            areasList.append(area['X'])
        return removeDuplicates(areasList)
    

    '''
    Metodo che restituisce la gravità dei crimini dell'area specificata
    Parametri:
    - areaNum (Int): il numero dell'area
    '''
    def getAreaSeverity(self, areaNum):
        severity = list(self.prolog.query(f"severity(area({areaNum}), X)"))
        return severity[0]['X']
    

    '''
    Funzione che restituisce la lista delle aree valutabili
    Parametri:
    - areas (List): la lista delle aree da cui selezionare le aree valutabili
    '''
    def evaluableAreas(self, areas):
        evAreas = list(self.prolog.query("isConsiderable(area(X))"))
        areasList = []
        for evArea in evAreas:
            if evArea['X'] in areas:
                areasList.append(evArea['X'])
        return areasList