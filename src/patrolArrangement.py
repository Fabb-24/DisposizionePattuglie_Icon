from icon.csp import optimizationCsp

class PatrolArrangement:
    """
    La classe PatrolArrangement rappresenta il problema di ottimizzazione per la disposizione delle pattuglie

    Attributes:
        kb (KB): La knowledge base da utilizzare
    """

    def __init__(self, kb):
        self.kb = kb

    def findBestArrangement(self, bound=float('inf')):
        """
        Funzione che risolve il problema di ottimizzazione tramite CSP e restituisce la miglior disposizoine delle pattuglie

        Parametri:
            kb (KB): la knowledge base da utilizzare
            bound (Int): il bound iniziale
        
        Returns:
            Dict: La migliore disposizione delle pattuglie
        """

        areaList = self.kb.getAreasList()

        dm = {}
        for area in areaList:
            dm[area] = [False, True]

        ocsp = optimizationCsp(variables=areaList,
                                constraints=areaList,
                                domains=dm, 
                                cost_function=self.cost, 
                                heuristic_function=self.h,
                                evaluableConstraints_function=self.evaluableAreas,
                                selectVariable_function=self.selectVariable,
                                bound=bound
        )

        return ocsp.solve()


    def cost(self, context, Cs):
        """
        Metodo che calcola il costo del contesto specificato in base ai vincoli che possono essere valutati.
        Se un vincolo non è soddisfatto, il costo del contesto è pari a infinito.
        Il costo è dato dal numero di pattuglie assegnate dove ogni pattuglia ha un peso diverso in base alla gravità dell'area

        Parametri:
            context (Dict): il contesto corrente
            Cs (List): la lista dei vincoli valutabili
        
        Returns:
            Float: Il costo del contesto specificato
        """

        for c in Cs:
            if not self.kb.isAreaSafe(c):
                return float('inf')
        
        areas = context.keys()
        cost = 0
        for area in areas:
            if context[area]:
                if self.kb.getAreaSeverity(area) == 2:
                    cost += 1
                elif self.kb.getAreaSeverity(area) == 1:
                    cost += 2
                else:
                    cost += 3
        return cost


    def h(self, Cs):
        """
        Funzione che restituisce l'euristica h per il contesto sp0ecificato sulla base dei vincoli non ancora soddisfatti

        Parametri:
            Cs (List): la lista dei vincoli rimanenti da soddisfare

        Returns:
            Int: L'euristica h per il contesto specificato
        """

        h = 0
        for c in Cs:
            if self.kb.getAreaSeverity(c) == 2:
                h += 1
        return h


    def selectVariable(self, Vs, context):
        """
        Funzione che seleziona la variabile da assegnare in base al contesto specificato e alle variabili rimanenti.
        Vengono preferite le variabili che sono vicine ad aree già assegnate e che hanno gravità minore

        Parametri:
            Vs (List): la lista delle variabili rimanenti
            context (Dict): il contesto corrente

        Returns:
            Any: La variabile da assegnare
        """

        bestArea = Vs[0]
        keys = list(context.keys())
        for area in keys:
            if context[area]:
                nearAreas = self.kb.getAreasByDistance(area, 1)
                for nearArea in nearAreas:
                    if nearArea in Vs:
                        if self.kb.getAreaSeverity(nearArea) == 0:
                            return nearArea
                        elif self.kb.getAreaSeverity(nearArea) == 1:
                            bestArea = nearArea
        return bestArea


    def evaluableAreas(self, CCs, context):
        """
        Funzione che restituisce la lista delle aree valutabili (di cui si possono valutare i vincoli)

        Parametri:
            CCs (List): la lista dei vincoli
            context (Dict): il contesto corrente

        Returns:
            List: La lista delle aree valutabili
        """

        keys = list(context.keys())
        areas = self.kb.getAreasList()

        for area in keys:
            self.kb.setAreaPatrol(area, context[area])
        
        for area in areas:
            if area not in keys:
                self.kb.removeAreaPatrol(area)
        
        can_eval = list(dict.fromkeys(self.kb.evaluableAreas(CCs)))

        return can_eval