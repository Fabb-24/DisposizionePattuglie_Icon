import os


def removeDuplicates(lst):
    """
    Funzione che rimuove i duplicati in una lista

    Parametri:
        lst (List): la lista da cui rimuovere i duplicati

    Returns:
        List: la lista senza duplicati
    """

    return list(dict.fromkeys(lst))


def removeElements(lst1, lst2):
    """
    Funzione che rimuove gli elementi della seconda lista dalla prima lista

    Parametri:
        lst1 (List): la lista da cui rimuovere gli elementi
        lst2 (List): la lista degli elementi da rimuovere

    Returns:
        List: la prima lista senza gli elementi della seconda lista
    """

    return [elem for elem in lst1 if elem not in lst2]


def printSolution(sol):
    """
    Metodo che stampa la soluzione del problema di CSP. Se la soluzione è None, stampa "No solution found", altrimenti stampa le aree da pattugliare

    Parametri:
        sol (Dict): la soluzione del problema di CSP
    """

    if sol is None:
        print("Nessuna soluzione trovata")
    else:
        print("Aree da pattugliare:", end=" ")
        for key in sol:
            if sol[key]:
                print(key, end=" ")


def getBasePath():
    """
    Metodo che ritorna il path base del progetto

    Returns:
        str: il path base del progetto
    """

    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "..")