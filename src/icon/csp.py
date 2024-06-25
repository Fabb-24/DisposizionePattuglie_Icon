from util import removeElements

__all__ = ['optimizationCsp']

class optimizationCsp:
    """
    La classe optimizationCsp rappresenta un problema di ottimizzazione tramite Constraint Satisfaction Problem (CSP).

    Attributes:
        variables (list): Lista delle variabili del problema\n
        constraints (list): Lista dei vincoli del problema\n
        domains (dict): Dizionario dei domini delle variabili del problema\n
        cost_function (function): Funzione che calcola il costo del contesto specificato in base ai vincoli che possono essere valutati\n
        heuristic_function (function): Funzione che restituisce l'euristica h per il contesto specificato sulla base dei vincoli non ancora soddisfatti\n
        evaluableConstraints_function (function): Funzione che restituisce la lista delle aree valutabili\n
        selectVariable_function (function): Funzione che seleziona la variabile da assegnare in base al contesto specificato e alle variabili rimanenti. Se non specificata, viene utilizzata la funzione di default che seleziona la prima variabile della lista delle variabili rimanenti\n
        bound (float): Limite superiore del costo del contesto
    """

    __all__ = ['solve']


    def __init__(self, variables, constraints, domains, cost_function, heuristic_function, evaluableConstraints_function, selectVariable_function=None, bound=float('inf')):
        self.Vs = variables
        self.Cs = constraints
        self.Ds = domains
        self.cost = cost_function
        self.h = heuristic_function
        self.evalCs = evaluableConstraints_function
        self.bound = bound
        self.best_asst = None
        if selectVariable_function is None:
            self.selectVariable = self.selectVariable_default
        else:
            self.selectVariable = selectVariable_function


    def solve(self):
        """
        Funzione di risoluzione del problema di CSP.
        Restituisce la migliore assegnazione delle variabili del problema.

        Returns:
            Dict: La migliore assegnazione delle variabili del problema
        """

        print("Inizio risoluzione CSP...")
        self.cbsearch(self.Vs, self.Cs, {})

        return self.best_asst


    def cbsearch(self, CVs, CCs, context):
        """
        Metodo ricorsivo della ricerca branch-and-bound per risolvere il problema di CSP

        Attributes:
            CVs (list): Lista delle variabili rimanenti da assegnare
            CCs (list): Lista dei vincoli rimanenti da soddisfare
            context (dict): Il contesto corrente
        """

        can_eval = self.evalCs(CCs, context)
        rem_Cs = CCs.copy()
        rem_Cs = removeElements(rem_Cs, can_eval)
        cost_context = self.cost(context, can_eval)

        if cost_context + self.h(rem_Cs) < self.bound:
            if not CVs:
                self.best_asst = context
                self.bound = cost_context
            else:
                var = self.selectVariable(CVs, context)
                for val in self.Ds[var]:
                    CVs2 = CVs.copy()
                    CVs2.remove(var)
                    context2 = context.copy()
                    context2[var] = val
                    self.cbsearch(CVs2, rem_Cs, context2)
    

    def selectVariable_default(self, CVs, context):
        """
        Funzione di default per la selezione della variabile da assegnare.
        Restituisce la prima variabile della lista delle variabili rimanenti.

        Attributes:
            CVs (list): Lista delle variabili rimanenti da assegnare
            context (dict): Il contesto corrente

        Returns:
            Any: La variabile da assegnare
        """

        return CVs[0]