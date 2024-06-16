from clearDataset import clearChicagoAreas, clearChicagoCrimes
from PrologKB import KB
from csp import CSP
from supervisedLearning import SupervisedLearning


'''
Metodo che stampa la soluzione del problema di CSP. Se la soluzione è None, stampa "No solution found", altrimenti stampa le aree da pattugliare
Parametri:
- sol (Dict): la soluzione del problema di CSP
'''
def printSolution(sol):
    if sol is None:
        print("No solution found")
    else:
        print("Areas to patrol:", end=" ")
        for key in sol:
            if sol[key]:
                print(key, end=" ")


if __name__ == "__main__":
    # Pulizia dei dataset
    chicagoAreasDf = clearChicagoAreas()
    chicagoCrimesDf = clearChicagoCrimes()

    # Apprendimento supervisionato per la previsione della gravità dei crimini
    supervisedLearning = SupervisedLearning(chicagoCrimesDf, "Severity")
    supervisedLearning.trainModel()

    '''# Creazione della knowledge base
    prolog = KB(chicagoAreasDf, "RandomForest", 26, 3, 2)

    # Risoluzione del problema di CSP
    csp = CSP(prolog, 150)
    sol = csp.solve()
    printSolution(sol)'''