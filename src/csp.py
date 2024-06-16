'''
Classe che rappresenta il problema di CSP per la disposizione delle pattuglie
'''
class CSP:
    '''
    Costruttore della classe. Inizializza il problema di CSP con la knowledge base e il bound iniziale specificati e inizializza la soluzione migliore inizialmente a None
    Parametri:
    - kb (KB): la knowledge base da utilizzare
    - bound (Int): il bound iniziale
    '''
    def __init__(self, kb, bound):
        self.kb = kb
        self.bound = bound
        self.best_asst = None
        
    
    '''
    Funzione che risolve il problema di CSP e restituisce la soluzione migliore
    '''
    def solve(self):
        # Le variabili del problema sono le aree di Chicago
        Vs = self.kb.getAreasList()
        # I vincoli del problema sono indicati tramite il numero dell'area su cui il vincolo viene applicato
        Cs = self.kb.getAreasList()

        print("Start solving CSP...")
        self.cbsearch(Vs, Cs, {})
        return self.best_asst
        

    '''
    Metodo ricorsivo della ricerca branch-and-bound per risolvere il problema di CSP
    Parametri:
    - CVs (List): la lista delle variabili rimanenti da assegnare
    - CCs (List): la lista dei vincoli rimanenti da soddisfare
    - context (Dict): il contesto corrente
    '''
    def cbsearch(self, CVs, CCs, context):
        can_eval = self.evaluableConstraints(CCs, context)
        rem_Cs = CCs.copy()
        for c in can_eval:
            rem_Cs.remove(c)
        cost_context = self.cost(context, can_eval)

        if cost_context + self.h(rem_Cs) < self.bound:
            #print(f"Cost: {cost_context}\tBound: {self.bound}")
            if not CVs:
                self.best_asst = context
                self.bound = cost_context
            else:
                var = self.selectVariable(CVs, context)
                for val in [False, True]:
                    if val:
                        self.kb.setAreaPatrol(var)
                    CVs2 = CVs.copy()
                    CVs2.remove(var)
                    context2 = context.copy()
                    context2[var] = val
                    self.cbsearch(CVs2, rem_Cs, context2)
                    self.kb.removeAreaPatrol(var)


    '''
    Metodo che restituisce i vincoli che possono essere valutati nel contesto specificato.
    Un vincolo può essere valutato se tutte le aree che rientrano nell'ambito del vincolo sono già state assegnate
    Parametri:
    - Cs (List): la lista dei vincoli da valutare
    - context (Dict): il contesto corrente
    '''
    def evaluableConstraints(self, Cs, context):
        can_eval = []
        keys = list(context.keys())
        for c in Cs:
            scope = self.scope(c)
            '''for area in scope:
                if area in keys and context[area]:
                    can_eval.append(c)'''
            if all(area in keys for area in scope) and c not in can_eval:
                can_eval.append(c)
        return can_eval


    '''
    Metodo che restituisce l'ambito del vincolo specificato.
    L'ambito di un vincolo è costituito dall'area specificata e dalle aree ad una distanza che varia in base alla gravità dell'area specificata
    Parametri:
    - c (Int): il numero dell'area su cui il vincolo è applicato
    '''
    def scope(self, c):
        gravity = self.kb.getAreaGravity(c)
        areas = self.kb.getAreasByDistance(c, 2 - gravity)
        areas.append(c)
        return areas


    '''
    Metodo che calcola il costo del contesto specificato in base ai vincoli che possono essere valutati.
    Se un vincolo non è soddisfatto, il costo del contesto è maggiore del bound.
    Il costo è dato dal numero di pattuglie assegnate dove ogni pattuglia ha un peso diverso in base alla gravità dell'area
    Parametri:
    - context (Dict): il contesto corrente
    - Cs (List): la lista dei vincoli che possono essere valutati
    '''
    def cost(self, context, Cs):
        for c in Cs:
            if self.constraintViolated(c):
                return self.bound + 1
        
        areas = context.keys()
        cost = 0
        for area in areas:
            if context[area]:
                if self.kb.getAreaGravity(area) == 2:
                    cost += 1
                elif self.kb.getAreaGravity(area) == 1:
                    cost += 2
                else:
                    cost += 3
        return cost
    

    '''
    Funzione che verifica se il vincolo specificato è violato
    Parametri:
    - c (Int): il numero dell'area su cui il vincolo è applicato
    '''
    def constraintViolated(self, c):
        return not self.kb.isAreaSecure(c)


    '''
    Funzione che restituisce l'euristica h per il contesto sp0ecificato sulla base dei vincoli non ancora soddisfatti
    Parametri:
    - context (Dict): il contesto corrente
    - Cs (List): la lista dei vincoli rimanenti da soddisfare
    '''
    def h(self, Cs):
        h = 0
        for c in Cs:
            if self.kb.getAreaGravity(c) == 2:
                h += 1
        return h


    '''
    Funzione che seleziona la variabile da assegnare in base al contesto specificato e alle variabili rimanenti.
    Vengono preferite le variabili che sono vicine ad aree già assegnate e che hanno gravità minore
    Parametri:
    - Vs (List): la lista delle variabili rimanenti
    - context (Dict): il contesto corrente
    '''
    def selectVariable(self, Vs, context):
        bestArea = Vs[0]
        keys = list(context.keys())
        for area in keys:
            if context[area]:
                nearAreas = self.kb.getAreasByDistance(area, 1)
                for nearArea in nearAreas:
                    if nearArea in Vs:
                        if self.kb.getAreaGravity(nearArea) == 0:
                            return nearArea
                        elif self.kb.getAreaGravity(nearArea) == 1:
                            bestArea = nearArea
        return bestArea