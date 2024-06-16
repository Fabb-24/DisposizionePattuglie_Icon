import pandas as pd
import geopandas as gpd
from shapely import wkt


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
def clearChicagoAreas():
    # Lettura del dataset
    df = pd.read_csv("dataset/chicagoAreas.csv")
    # Rimozione delle colonne inutili
    df = df.drop(columns=["perimeter", "area", "comarea_", "comarea_id", "area_numbe", "shape_area", "shape_len"])
    # Rinomina delle colonne
    df = df.rename(columns={"the_geom": "Perimeter", "community": "AreaName", "area_num_1": "AreaNumber"})
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
    medium_severity = ['assault', 'battery', 'burglary', 'weapons violation', 'narcotics', 'motor vehicle theft', 'criminal damage', 'offense involving children', 'prostitution', 'stalking']
    low_severity = ['theft', 'other offense', 'deceptive practice', 'criminal trespass', 'interference with public officer', 'public peace violation', 'gambling', 'intimidation', 'obscenity', 'non-criminal', 'liquor law violation', 'public indecency', 'ritualism', 'crim sexual assault', 'other narcotic violation', 'concealed carry license violation']
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
def clearChicagoCrimes():
    # Lettura del dataset
    df = pd.read_csv("dataset/chicagoCrimes2020.csv")
    # Rimozione delle colonne inutili
    df = df.drop(columns=["ID", "Case Number", "Block", "IUCR", "Description", "Location Description", "Arrest", "Domestic", "Beat", "District", "Ward", "FBI Code", "X Coordinate", "Y Coordinate", "Year", "Updated On", "Latitude", "Longitude", "Location"])
    # Rimozione dei crimini nelle aree non utilizzate
    areas = getChicagoAreas()
    df = df[df['Community Area'].isin(areas)]
    # Eliminazione delle righe con valori nulli
    df = df.dropna()
    # Sostituzione della colonna Date con le colonne Week, Day e Time Slot
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.dayofweek
    df['Time Slot'] = df['Date'].dt.hour
    df.drop(columns=['Date'], inplace=True)
    # Conversione della colonna Week in intero
    df['Week'] = df['Week'].astype(int)
    # Conversione della colonna Community Area in intero
    df['Community Area'] = df['Community Area'].astype(int)
    # Conversione dei tipi di crimine in base alla loro gravità e rinomina della colonna
    crimeSeverity = divideCrimeType()
    df['Primary Type'] = df['Primary Type'].str.lower().map(crimeSeverity)
    df = df.rename(columns={"Primary Type": "Severity"})
    # Colonne rimaste: Severity, Community Area, Week, Day, Time Slot

    return df