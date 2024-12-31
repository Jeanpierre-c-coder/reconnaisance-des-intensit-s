import numpy as np

def load_events(filename):
    """
    Charge les événements à partir d'un fichier.
    
    Args:
        filename (str): Chemin du fichier texte contenant les événements.
    
    Returns:
        dict: Dictionnaire contenant t, x, y et p comme clés.
    """
    events = {"t": [], "x": [], "y": [], "p": []}
    with open(filename, 'r') as file:
        for line in file:
            timestamp, x, y, polarity = map(float, line.strip().split())
            events["t"].append(timestamp)
            events["x"].append(int(x))
            events["y"].append(int(y))
            events["p"].append(int(polarity))
    return events

durée du fichier
events = load_events("short_shapes.txt")
duration = max(events["t"]) - min(events["t"])
print(f"Durée totale : {duration:.2f} s")

nombre de fichier
total_events = len(events["t"])
print(f"Nombre total d'événements : {total_events}")


import matplotlib.pyplot as plt
import numpy as np

def disp_video(events, timewindow):
    """
    Affiche une vidéo des événements sur un fond gris.
    
    Args:
        events (dict): Dictionnaire des événements.
        timewindow (float): Intervalle de temps entre deux affichages.
    """
    t_min = min(events["t"])
    t_max = max(events["t"])
    current_time = t_min
    
    while current_time < t_max:
        # Filtrer les événements dans la fenêtre temporelle
        mask = (events["t"] >= current_time) & (events["t"] < current_time + timewindow)
        x, y, p = np.array(events["x"])[mask], np.array(events["y"])[mask], np.array(events["p"])[mask]
        
        # Créer une image
        image = np.full((max_y + 1, max_x + 1), 0.5)  # Fond gris
        image[x, y] = np.where(p == 1, 1.0, 0.0)  # Blanc pour ON, Noir pour OFF
        
        # Afficher l'image
        plt.imshow(image, cmap="gray")
        plt.title(f"Temps : {current_time:.2f} s")
        plt.pause(0.1)
        current_time += timewindow
