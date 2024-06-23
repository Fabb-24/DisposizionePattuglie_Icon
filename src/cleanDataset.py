import pandas as pd
import geopandas as gpd
from shapely import wkt
import os
from util import getBasePath


'''
Funzione che restituisce le aree di Chicago che verranno utilizzate
'''
def getChicagoAreas():
    areas = [5, 6, 7, 21, 22, #north side
            15, 16, 17, 18, 19, 20, #northwest side
            8, 32, 33, #central
            23, 24, 25, 26, 27, 28, 29, 30, 31, #west side
            1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 76, 77 #far north side
            ]
    return areas


'''
Funzione che sistema il dataset che contiene le aree di Chicago e lo restituisce
'''
def cleanChicagoAreas():
    dfPath = os.path.join(getBasePath(), "dataset", "chicagoAreas.csv")
    # Lettura del dataset
    df = pd.read_csv(dfPath)
    # Rimozione delle colonne inutili
    df = df.drop(columns=["perimeter", "area", "comarea_", "comarea_id", "area_numbe", "shape_len"])
    # Rinomina delle colonne
    df = df.rename(columns={"the_geom": "Perimeter", "community": "AreaName", "area_num_1": "AreaNumber", "shape_area": "AreaSize"})
    # Converti la colonna contenente la geometria dell'area in una geometria utilizzabile
    df['Perimeter'] = df['Perimeter'].apply(wkt.loads)
    # Crea un GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='Perimeter')
    # Rimozione delle aree non utilizzate
    areas = getChicagoAreas()
    gdf = gdf[gdf['AreaNumber'].isin(areas)]

    # Colonne rimaste: AreaNumber, AreaName, Perimeter
    return gdf


'''
Funzione che divide i tipi di crimine in base alla loro gravità
'''
def divideCrimeType():
    high_severity = ['homicide', 'kidnapping', 'criminal sexual assault', 'arson', 'robbery', 'sex offense', 'human trafficking']
    medium_severity = ['theft', 'assault', 'battery', 'burglary', 'weapons violation', 'narcotics', 'motor vehicle theft', 'criminal damage', 'offense involving children', 'prostitution', 'stalking', 'crim sexual assault']
    low_severity = ['theft', 'other offense', 'deceptive practice', 'criminal trespass', 'interference with public officer', 'public peace violation', 'gambling', 'intimidation', 'obscenity', 'non-criminal', 'liquor law violation', 'public indecency', 'ritualism', 'other narcotic violation', 'concealed carry license violation']
    # Mappatura dei tipi di crimine con la loro gravità (0 = bassa, 1 = media, 2 = alta)
    crimeSeverity = {}
    for crime in high_severity:
        crimeSeverity[crime] = 2
    for crime in medium_severity:
        crimeSeverity[crime] = 1
    for crime in low_severity:
        crimeSeverity[crime] = 0

    return crimeSeverity


'''
Funzione che sistema il dataset che contiene i crimini di Chicago e lo restituisce
'''
def cleanChicagoCrimes():
    dfPath = os.path.join(getBasePath(), "dataset", "chicagoCrimes2018.csv")
    # Lettura del dataset
    df = pd.read_csv(dfPath)
    # Rimozione delle colonne inutili
    df = df.drop(columns=["ID", "Case Number", "Block", "IUCR", "Description", "Beat", "Location Description", "District", "Ward", "FBI Code", "X Coordinate", "Y Coordinate", "Year", "Updated On", "Latitude", "Longitude", "Location"])
    # Rimozione dei crimini nelle aree non utilizzate
    areas = getChicagoAreas()
    df = df[df['Community Area'].isin(areas)]
    # Eliminazione delle righe con valori nulli
    df = df.dropna()
    # Sostituzione della colonna Date con le colonne Week, Day e Time Slot
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.dayofweek
    #df['Time Slot'] = df['Date'].dt.hour
    #divisione in fasce orarie nel formato HHmm dove mm è 00 o 30
    #df['Time Slot'] = df['Date'].dt.hour * 100 + (df['Date'].dt.minute // 30) * 30
    df['Time Slot'] = df['Date'].dt.hour
    df.drop(columns=['Date'], inplace=True)
    # Conversione della colonna Week in intero
    df['Week'] = df['Week'].astype(int)
    # Conversione della colonna Community Area in intero
    df['Community Area'] = df['Community Area'].astype(int)
    # Conversione dei tipi di crimine in base alla loro gravità e rinomina della colonna
    crimeSeverity = divideCrimeType()
    df['Primary Type'] = df['Primary Type'].str.lower().map(crimeSeverity)
    df = df.rename(columns={"Primary Type": "Crime Severity"})

    # Converto le colonne Arrest e Domestic in interi (0 = False, 2 = True)
    df['Arrest'] = df['Arrest'].astype(int)*2
    df['Domestic'] = df['Domestic'].astype(int)*2

    # Creo la colonna "Severity" che è la media tra "Crime Severity", "Arrest" e "Domestic", arrotondata normalmente all'intero più vicino
    df['Severity'] = df[['Crime Severity', 'Arrest', 'Domestic']].mean(axis=1).round().astype(int)
    # Rimozione delle colonne inutili
    df = df.drop(columns=["Crime Severity", "Arrest", "Domestic"])

    # Colonne rimaste: Severity, Community Area, Week, Day, Time Slot
    return df