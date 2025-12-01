import streamlit as st
import pandas as pd
from Vitrage_2 import *
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as xl
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

#-----------------------Importation des points d'origine----------------------"
def CoordPoint(doc,feuille):
    document =xl.load_workbook(doc,data_only=True)
    sheet = document[feuille]
    max_row = sheet.max_row
    Coord = {}
    Depl = {}
    labels = []
    for row in range(4, max_row+1):
        label = sheet.cell(row, column=3).value
        labels.append(label)
        
    #Attribuer les données
    i = 0
    for row in range(4, max_row+1) :
        X_Origin = sheet.cell(row, column=4).value
        Y_Origin = sheet.cell(row, column=5).value
        Z_Origin = sheet.cell(row, column=6).value
        coord_origin = [X_Origin,Y_Origin,Z_Origin]
        Coord[labels[i]] = coord_origin
        
        X_Depl = sheet.cell(row, column=7).value 
        Y_Depl = sheet.cell(row, column=8).value 
        Z_Depl = sheet.cell(row, column=9).value
        deplacement = [X_Depl,Y_Depl,Z_Depl]
        Depl[labels[i]] = deplacement
        i = 1+i
        
    return Coord, Depl

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

#-------------------------------changement de repére--------------------------

def rot_repere(angle,point):
    y = cos(radians(-angle))*point[1]-sin(radians(-angle))*point[2]
    z = sin(radians(-angle))*point[1]+cos(radians(-angle))*point[2]
    return[point[0], y, z]

#---------------------------Chargement des vitrages---------------------------"
def Dechaussement_vitrage(uploaded_file, option_calage, Raico):
    Feuille_coord = 'Coord_Points'
    Coord,Depl = CoordPoint(uploaded_file,Feuille_coord)
    
    document =xl.load_workbook(uploaded_file,data_only=True)
    sheet = document['Vitrages']
    max_row = sheet.max_row
    datas = {
        'ID vitrage': [],
        'Gamme': [],
        'Demi-périmètre (m)': [],
        'Ecart minimal // bornes gauche (mm)': [],
        'Ecart minimal // bornes droite (mm)': [],
        'Ecart minimal // bornes hautes (mm)': []
    }

    graphs = {
        'ID vitrage': [],
        'Graph': [],
    }

    for row in range(3, max_row+1) :
        if sheet.cell(row+1, column=2).value == None:
            break
        
        labels = {
            "A": sheet.cell(row + 1, column=2).value,
            "B": sheet.cell(row + 1, column=3).value,
            "C": sheet.cell(row + 1, column=4).value,
            "D": sheet.cell(row + 1, column=5).value,
        }
        Gamme = sheet.cell(row + 1, column=6).value        
        
        cadre_def = {}
        cadre_0 = {}
        
        for key, label in labels.items():
            print([label][0],[labels["D"]][0])
            x = Coord[label][0] + Depl[label][0] - (Coord[labels["D"]][0] + Depl[labels["D"]][0])
            if key == "D":  # cas particulier pour D
                x = 0

            if key == "A":
                y = Coord[label][1] - Depl[labels["D"]][1] - (Coord[labels["D"]][1] - Depl[labels["D"]][1])
            elif key == "B":
                y = Coord[label][1] - Depl[labels["C"]][1] - (Coord[labels["D"]][1] - Depl[labels["D"]][1])
            elif key == "C":
                y = Coord[label][1] - Depl[labels["C"]][1] - (Coord[labels["D"]][1] - Depl[labels["D"]][1])
            else:  # D
                y = Coord[label][1] - Depl[label][1] - (Coord[labels["D"]][1] - Depl[labels["D"]][1])

            z = Coord[label][2] + Depl[label][2]

            cadre_def[key] = [x, y, z]

            x_0 = Coord[label][0] - Coord[labels["D"]][0]
            y_0 = Coord[label][1] - Coord[labels["D"]][1]
            z_0 = Coord[label][2]

            cadre_0[key] = [x_0, y_0, z_0]
        
        angle = degrees(arctan(cadre_0['A'][2]/cadre_0['A'][1]))
        print(angle)

        for key, coords in cadre_def.items():
             cadre_def[key] = rot_repere(angle,coords)

        for key, coords in cadre_0.items():
             cadre_0[key] = rot_repere(angle,coords)

        H = abs(cadre_0['A'][1]-cadre_0['D'][1])
        L = abs(cadre_0['A'][0]-cadre_0['B'][0])
        diag = sqrt(H**2+L**2)
        pf = L/1000 + H/1000
        V = Vitrage(cadre_0,cadre_def,Gamme,Raico,pf,calage_lateral=option_calage,)
        data, graph = V.Dechaussement()

        datas['ID vitrage'].append(str(labels["A"])+' / '+str(labels["B"])+' / '+str(labels["C"])+' / '+str(labels["D"]))
        datas['Gamme'].append(Gamme)
        datas['Demi-périmètre (m)'].append(round(pf,1))
        datas['Ecart minimal // bornes gauche (mm)'].append(data[0])
        datas['Ecart minimal // bornes droite (mm)'].append(data[1])
        datas['Ecart minimal // bornes hautes (mm)'].append(data[2])

        graphs['ID vitrage'].append(str(labels["A"])+' / '+str(labels["B"])+' / '+str(labels["C"])+' / '+str(labels["D"]))
        graphs['Graph'].append(graph)
        
    return datas, graphs

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-----------------------visualiser les quadrilatères avec zoom----------------

def visualiser_quadrilateres(quadrilateres, titre_page=None):
    """Crée deux graphiques avec zoom adaptatif sur les coins supérieurs"""
    # Format A4 en paysage : 11.69 x 8.27 pouces
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    
    # Titre de la page si fourni
    if titre_page:
        fig.suptitle(titre_page, fontsize=16, fontweight='bold', y=0.98)
    
    couleurs =["#000000","#FF0000", '#006D8F', "#FF0000"]
    titre_graph =['Cadre','Limite','Vitrage','Limite']
    
    # Tracer les quadrilatères sur les deux axes - contours uniquement
    for ax in [ax1, ax2]:
        for i, quad in enumerate(quadrilateres):
            x, y = quad.exterior.xy
            patch = MplPolygon(list(zip(x, y)), facecolor='none', 
                             edgecolor=couleurs[i], linewidth=1.0,
                             label=titre_graph[i])
            ax.add_patch(patch)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Zoom adaptatif sur le coin supérieur gauche
    quad1 = quadrilateres[2]
    x1, y1 = quad1.exterior.xy
    
    # Trouver le coin supérieur gauche (point avec x min et y max)
    points1 = list(zip(x1, y1))
    coin_haut_gauche = min(points1, key=lambda p: p[0] - p[1])  # x minimum, y maximum
    
    # Calculer la taille de la zone de zoom basée sur la taille du quadrilatère
    largeur_quad1 = max(x1) - min(x1)
    hauteur_quad1 = max(y1) - min(y1)
    marge = max(largeur_quad1, hauteur_quad1) * 0.01  # 1% de marge
    
    ax1.set_xlim(coin_haut_gauche[0] - marge, coin_haut_gauche[0] + marge)
    ax1.set_ylim(coin_haut_gauche[1] - marge, coin_haut_gauche[1] + marge)
    ax1.set_title("Zoom - Coin Supérieur Gauche", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    
    # Zoom adaptatif sur le coin supérieur droit
    quad4 = quadrilateres[2]
    x4, y4 = quad4.exterior.xy
    
    # Trouver le coin supérieur droit (point avec x max et y max)
    points4 = list(zip(x4, y4))
    coin_haut_droit = max(points4, key=lambda p: p[0] + p[1])  # x maximum, y maximum
    
    # Calculer la taille de la zone de zoom basée sur la taille du quadrilatère
    largeur_quad4 = max(x4) - min(x4)
    hauteur_quad4 = max(y4) - min(y4)
    marge = max(largeur_quad4, hauteur_quad4) * 0.01  # 1% de marge
    
    ax2.set_xlim(coin_haut_droit[0] - marge, coin_haut_droit[0] + marge)
    ax2.set_ylim(coin_haut_droit[1] - marge, coin_haut_droit[1] + marge)
    ax2.set_title("Zoom - Coin Supérieur Droit", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    
    plt.tight_layout()
    return fig

#----------------------Edition du rapport pdf FORMAT A4 PAYSAGE---------------

def creer_page_complete(ligne_data, quadrilateres):
    """Crée une page A4 PAYSAGE avec les détails et les deux graphiques côte à côte"""
    # Format A4 en PAYSAGE : 11.69 x 8.27 pouces
    fig = plt.figure(figsize=(11.69, 8.27))
    
    # Titre de la page
    fig.suptitle(f"Vitrage {ligne_data['ID vitrage']}", 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Créer une grille pour organiser le contenu en PAYSAGE
    # 3 lignes, 2 colonnes : [Métriques sur toute la largeur] [Graph gauche | Graph droit]
    gs = fig.add_gridspec(3, 2, height_ratios=[0.25, 0.6, 0.15],hspace=0.3, wspace=0.20,
                          left=0.06, right=0.97, top=0.90, bottom=0.06)
    
    # Section 1: Tableau des métriques principales (en haut, sur toute la largeur)
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_metrics.axis('tight')
    ax_metrics.axis('off')

   # Récupérer les valeurs des écarts
    ecart_gauche = ligne_data['Ecart minimal // bornes gauche (mm)']
    ecart_droite = ligne_data['Ecart minimal // bornes droite (mm)']
    ecart_hautes = ligne_data['Ecart minimal // bornes hautes (mm)']

    metrics_data = [
        ['Paramètre', 'Valeur', 'Unité'],
        ['Gamme', f"{ligne_data['Gamme']}", '-'],
        ['Demi-périmètre', f"{ligne_data['Demi-périmètre (m)']:.1f}", 'm'],
        ['Ecart minimal // bornes gauche', f"{ligne_data['Ecart minimal // bornes gauche (mm)']:.1f}", 'mm'],
        ['Ecart minimal // bornes droite', f"{ligne_data['Ecart minimal // bornes droite (mm)']:.1f}", 'mm'],
        ['Ecart minimal // bornes hautes', f"{ligne_data['Ecart minimal // bornes hautes (mm)']:.1f}", 'mm']
    ]
    
    table = ax_metrics.table(cellText=metrics_data, cellLoc='center', loc='center',
                            colWidths=[0.3, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style du tableau
    # Style du tableau avec code couleur pour les écarts
    for i in range(len(metrics_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # En-tête
                cell.set_facecolor("#006D8F")
                cell.set_text_props(weight='bold', color='white')
            elif i == 3 and j == 1:  # Ecart gauche
                if ecart_gauche < 0:
                    cell.set_facecolor('#FFB3B3')  # Rouge clair
                else:
                    cell.set_facecolor('#B3FFB3')  # Vert clair
            elif i == 4 and j == 1:  # Ecart droite
                if ecart_droite < 0:
                    cell.set_facecolor('#FFB3B3')  # Rouge clair
                else:
                    cell.set_facecolor('#B3FFB3')  # Vert clair
            elif i == 5 and j == 1:  # Ecart hautes
                if ecart_hautes < 0:
                    cell.set_facecolor('#FFB3B3')  # Rouge clair
                else:
                    cell.set_facecolor('#B3FFB3')  # Vert clair
            else:
                cell.set_facecolor('#E7E6E6' if i == 1 or i == 2 else 'white')

    couleurs = ["#000000","#FF3333", '#006D8F', "#FF3333"]
    titre_graph = ['Cadre','Limite','Vitrage','Limite']
    
    # Section 3: GRAPHIQUE GAUCHE - Zoom coin supérieur gauche
    ax1 = fig.add_subplot(gs[1, 0])
    
    for i, quad in enumerate(quadrilateres):
        x, y = quad.exterior.xy
        patch = MplPolygon(list(zip(x, y)), facecolor='none', 
                         edgecolor=couleurs[i], linewidth=1.5,
                         label=titre_graph[i])
        ax1.add_patch(patch)
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Zoom adaptatif sur le coin supérieur gauche
    quad1 = quadrilateres[2]
    x1, y1 = quad1.exterior.xy
    points1 = list(zip(x1, y1))
    coin_haut_gauche = min(points1, key=lambda p: p[0] - p[1])
    
    largeur_quad1 = max(x1) - min(x1)
    hauteur_quad1 = max(y1) - min(y1)
    marge = max(largeur_quad1, hauteur_quad1) * 0.01
    
    ax1.set_xlim(coin_haut_gauche[0] - marge, coin_haut_gauche[0] + marge)
    ax1.set_ylim(coin_haut_gauche[1] - marge, coin_haut_gauche[1] + marge)
    ax1.set_title("Coin Supérieur Gauche", 
                  fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("Coordonnée X (mm)", fontsize=9)
    ax1.set_ylabel("Coordonnée Y (mm)", fontsize=9)
    ax1.tick_params(labelsize=8)
    
    # Section 4: GRAPHIQUE DROIT - Zoom coin supérieur droit
    ax2 = fig.add_subplot(gs[1, 1])
    
    for i, quad in enumerate(quadrilateres):
        x, y = quad.exterior.xy
        patch = MplPolygon(list(zip(x, y)), facecolor='none', 
                         edgecolor=couleurs[i], linewidth=1.5,
                         label=titre_graph[i])
        ax2.add_patch(patch)
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Zoom adaptatif sur le coin supérieur droit
    quad4 = quadrilateres[2]
    x4, y4 = quad4.exterior.xy
    points4 = list(zip(x4, y4))
    coin_haut_droit = max(points4, key=lambda p: p[0] + p[1])
    
    largeur_quad4 = max(x4) - min(x4)
    hauteur_quad4 = max(y4) - min(y4)
    marge = max(largeur_quad4, hauteur_quad4) * 0.01
    
    ax2.set_xlim(coin_haut_droit[0] - marge, coin_haut_droit[0] + marge)
    ax2.set_ylim(coin_haut_droit[1] - marge, coin_haut_droit[1] + marge)
    ax2.set_title("Coin Supérieur Droit", 
                  fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("Coordonnée X (mm)", fontsize=9)
    ax2.set_ylabel("Coordonnée Y (mm)", fontsize=9)
    ax2.tick_params(labelsize=8)

    # Section 2: Annotation (sur toute la largeur)
    ax_annotation = fig.add_subplot(gs[2, :])
    ax_annotation.axis('off')
    
    cadre = quadrilateres[0]
    A_c, B_c, C_c, D_c, origine_c = cadre.exterior.coords
    
    if D_c[1] > C_c[1]:
        Sens_y = 'droite'
        opp = 'gauche'
    else:
        Sens_y = 'gauche'
        opp = 'doite'
    if A_c[0] > D_c[0]:
        Sens_x = 'droite'
    else:
        Sens_x = 'gauche'

    if round(max(abs(A_c[0]-D_c[0]),abs(B_c[0]-C_c[0])),1) == 0:
        annotation_text = f"""Le montant de {Sens_y} descend de {round(abs(D_c[1]-C_c[1]),1)}mm par rapport au montant de {opp}.\n"""
    elif round(abs(D_c[1]-C_c[1]),1) == 0:
        annotation_text = f""" La traverse haute se déplace de {round(max(abs(A_c[0]-D_c[0]),abs(B_c[0]-C_c[0])),1)}mm vers la {Sens_x} par rapport à la traverse basse."""
    else:
        annotation_text = f"""Le montant de {Sens_y} descend de {round(abs(D_c[1]-C_c[1]),1)}mm par rapport au montant de {opp}.\n La traverse haute se déplace de {round(max(abs(A_c[0]-D_c[0]),abs(B_c[0]-C_c[0])),1)}mm vers la {Sens_x} par rapport à la traverse basse."""

    ax_annotation.text(0.5, 0.5, annotation_text, 
                      ha='center', va='center', 
                      fontsize=9, family='Sans',
                      bbox=dict(boxstyle='square,pad=0.3', facecolor="#EDF4F5", 
                               edgecolor="#006D8F", linewidth=1, alpha=0.9))
     
    return fig

# Fonction pour générer le rapport PDF complet
def generer_rapport_pdf(df, graphs):
    """Génère un rapport PDF avec toutes les lignes au format A4 PAYSAGE"""
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page de titre - Format A4 PAYSAGE
        fig_titre = plt.figure(figsize=(11.69, 8.27))
        
        # En-tête institutionnel
        fig_titre.text(0.5, 0.65, "ANALYSE DE LA MISE EN PARALLELOGRAMME\n DES VITRAGES", 
                       ha='center', va='center', 
                       fontsize=16, fontweight='bold', family='Sans')
        
        # Ligne de séparation
        fig_titre.text(0.5, 0.60, "─" * 80, 
                       ha='center', va='center', 
                       fontsize=10)

        #Mettre une image
        img = mpimg.imread("logo-couleur.png")
        ax = fig_titre.add_axes([0.38, 0.43, 0.25, 0.1375])  # [left, bottom, width, height]
        ax.imshow(img)
        ax.axis("off")

        # Résumer
        mask = (df[["Ecart minimal // bornes gauche (mm)", "Ecart minimal // bornes droite (mm)", "Ecart minimal // bornes hautes (mm)"]] < 0).any(axis=1)
        nb_non_conforme = mask.sum()
        nb_conforme = len(df)-nb_non_conforme
        
        Gammes = df["Gamme"].value_counts().index
        i=0
        for g in Gammes:
            if i==0:
                Gamme = str(g)
            else :
                Gamme += f" & {g}"
            i = i+1
        
        print(Gammes)
        stats_text = f"""
Gamme Raico de l'étude: {Gamme}
Nombre vitrage étudiés: {len(df)}
Nombre de vitrage conforme: {nb_conforme}
Nombre de vitrage non-conforme: {nb_non_conforme}
        """
        
        fig_titre.text(0.5, 0.35, stats_text, 
                       ha='center', va='center', 
                       fontsize=10, family='Sans', style='normal',
                       bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.8))
        
        # Pied de page
        from datetime import datetime
        date_rapport = datetime.now().strftime("%d/%m/%Y")
        fig_titre.text(0.5, 0.25, f"Date du rapport: {date_rapport}", 
                       ha='center', va='center', 
                       fontsize=11, family='Sans')

        fig_titre.text(0.5, 0.15, "MTECHBUILD", 
                       ha='center', va='center', 
                       fontsize=10, style='italic', color='gray', family='Sans')
        
        plt.axis('off')
        pdf.savefig(fig_titre, bbox_inches='tight')
        plt.close(fig_titre)
        
        # Pour chaque ligne - une page A4 PAYSAGE avec les deux graphiques côte à côte
        for idx, row in df.iterrows():
            quadrilateres = graphs["Graph"][idx]
            fig_complete = creer_page_complete(row, quadrilateres)
            pdf.savefig(fig_complete, bbox_inches='tight')
            plt.close(fig_complete)
        
        # Métadonnées du PDF
        d = pdf.infodict()
        d['Title'] = 'Analyse de la mise en paraléllogramme des vitrages'
        d['Author'] = 'MTECHBUILD'
        d['Subject'] = 'Analyse de la mise en paraléllogramme des vitrages'
        d['Keywords'] = 'Vitrage, Analyse, Paraléllogramme'
        d['Creator'] = 'Application Streamlit'
    
    buffer.seek(0)
    return buffer

def formater_metric(label, valeur):
    if valeur > 0:
        delta = '🟢 OK'
    elif valeur < 0:
        delta = '🔴 NOK'
    else:
        delta = '🟠 Calé latéralement'
    st.metric(label=label, value=valeur, delta=delta)

def colorer_valeurs(val):
    if val > 0:
        color = "green"
    elif val < 0:
        color = "red"
    else:
        color = "orange"
    return f"color: {color}; font-weight: bold;"
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Configuration de la page
st.set_page_config(page_title="Analyse de la mise en parallélogramme", layout="wide")

# Titre de l'application
st.title("🏢 Analyse de la mise en parallélogramme des vitrages")

choix = st.selectbox("Calage latéral :", ["Sans", "Avec"])

# --- Ligne 1 ---
ligne1_col1, ligne1_col2 = st.columns(2)

with ligne1_col1:
    if choix == "Sans":
        st.image("Borne_M.png", caption="Bornes sur montants", width=450)
    elif choix == "Avec":
        st.image("Borne_M_C.png", caption="Bornes sur montants", width=450)


with ligne1_col2:
    if choix == "Sans":
        st.image("Borne_T.png", caption="Bornes sur traverses", width=400)
    elif choix == "Avec":
        st.image("Borne_T_C.png", caption="Bornes sur traverses", width=400)   

# --- Ligne 2 ---
ligne2_col1, ligne2_col2 = st.columns(2)

with ligne2_col1:
    if choix == "Sans":
        Tm1 = st.number_input("Valeur Tolérance Tm1 :", value=2.5, step=0.1, format="%.2f")
        Tm2 = st.number_input("Valeur Tolérance Tm2 + menuiserie :", value=10.5, step=0.1, format="%.2f")
    else:
        Tm =  st.number_input("Valeur Tolérance Tm :", value=2.5, step=0.1, format="%.2f")
        Ca =  st.number_input("Epaisseur cale C + menuiserie :", value=13.0, step=0.1, format="%.2f")
        Jc =  st.number_input("Jeu entre vitrage et cale Jc :", value=2.0, step=0.1, format="%.2f")
    
with ligne2_col2:
    if choix == "Sans":
        Tt1 = st.number_input("Valeur Tolérance Tt1 :", value=5.0, step=0.1, format="%.2f")
        Tt2 = st.number_input("Valeur Tolérance Tt2 + menuiserie :", value=13.0, step=0.1, format="%.2f")
    else:
        Tt1 =  st.number_input("Valeur Tolérance Tt1 :", value=5.0, step=0.1, format="%.2f")
        Tt2 =  st.number_input("Valeur Tolérance Tt2 + menuiserie :", value=13.0, step=0.1, format="%.2f")
        Jv =  st.number_input("Jeu entre borne basse et vitrage Jv :", value=1.0, step=0.1, format="%.2f")


#---------------------------Importation des données---------------------------

st.header("📂 Importer Fichier Excel")

uploaded_file = st.file_uploader("Importer un fichier de données (CSV ou Excel)", type=["csv", "xlsx"])

#---------------------------Chargement des vitrages---------------------------
if choix == "Sans":
    Raico = [Tm1, Tm2, Tt1, Tt2]

else:
    Raico = [Tm, Ca, Jc, Tt1, Tt2, Jv]


if uploaded_file:
    # Mémoriser le nom du fichier pour savoir si un nouveau fichier est importé
    if "uploaded_name" not in st.session_state or st.session_state["uploaded_name"] != uploaded_file.name:
        st.session_state.clear()
        st.session_state["uploaded_name"] = uploaded_file.name

    # Vérifier si un changement de paramètre nécessite une mise à jour
    params_actuels = {
        "choix": choix,
        "Raico": Raico,
    }

    if (
        "params_prec" not in st.session_state
        or st.session_state["params_prec"] != params_actuels
    ):
        datas, graphs = Dechaussement_vitrage(uploaded_file, choix, Raico)
        st.session_state["datas"] = datas
        st.session_state["graphs"] = graphs
        st.session_state["params_prec"] = params_actuels
    else:
        datas = st.session_state["datas"]
        graphs = st.session_state["graphs"]

    # -------------------- Affichage du tableau --------------------
    st.header("📊 Tableau des Résultats")
    df_affichage = pd.DataFrame(datas)
    
    styled_df = (
    df_affichage.style
    .map(colorer_valeurs, subset=["Ecart minimal // bornes gauche (mm)", 
                                  "Ecart minimal // bornes droite (mm)", 
                                  "Ecart minimal // bornes hautes (mm)"])
    .format(precision=1)
    .set_properties(**{'text-align': 'center'})
    .set_properties(**{'text-align': 'center', 'vertical-align': 'middle'})
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]}
    ])
)
    st.dataframe(styled_df, width='stretch', hide_index=True)

    # -------------------- Sélection et visualisation --------------------
    st.header("🔲 Sélection et Visualisation")
    col1, col2 = st.columns([1, 3])

    with col1:
        ligne_selectionnee = st.selectbox(
            "Sélectionnez le vitrage à visualiser :",
            options=list(datas["ID vitrage"]),
            key="select_vitrage",
        )

        idx = datas["ID vitrage"].index(ligne_selectionnee)
        ligne_data = {col: datas[col][idx] for col in datas.keys()}

        for col in ligne_data.keys():
                if col != 'ID vitrage':
                    st.metric(col, ligne_data[col])

    with col2:
        idx = graphs["ID vitrage"].index(ligne_selectionnee)
        quadrilateres = graphs["Graph"][idx]
        fig = visualiser_quadrilateres(quadrilateres)
        st.pyplot(fig)
        # Informations supplémentaires
        cadre = graphs["Graph"][idx][0]
        A_c,B_c,C_c,D_c,origine_c = cadre.exterior.coords
        if D_c[1]>C_c[1]:
            Sens_y = 'droite'
        else:
            Sens_y = 'gauche'
        if A_c[0]>D_c[0]:
            Sens_x = 'droite'
        else:
            Sens_x = 'gauche'
        

        cadre = quadrilateres[0]
    A_c, B_c, C_c, D_c, origine_c = cadre.exterior.coords
    
    if D_c[1] > C_c[1]:
        Sens_y = 'droite'
        opp = 'gauche'
    else:
        Sens_y = 'gauche'
        opp = 'droite'
    if A_c[0] > D_c[0]:
        Sens_x = 'droite'
    else:
        Sens_x = 'gauche'

    if round(max(abs(A_c[0]-D_c[0]),abs(B_c[0]-C_c[0])),1) == 0:
        st.info(f"""📐
        Le montant de **{Sens_y}** descend de **{round(abs(D_c[1]-C_c[1]),1)}mm** par rapport au montant de {opp}.
        """)
    elif round(abs(D_c[1]-C_c[1]),1) == 0:
        st.info(f"""📐
        La traverse haute se déplace de **{round(max(abs(A_c[0]-D_c[0]),abs(B_c[0]-C_c[0])),1)}mm** vers la **{Sens_x}** par rapport à la traverse basse.
        """)
    else:
        st.info(f"""📐
        Le montant de **{Sens_y}** descend de **{round(abs(D_c[1]-C_c[1]),1)}mm** par rapport au montant de {opp}.\n La traverse haute se déplace de **{round(max(abs(A_c[0]-D_c[0]),abs(B_c[0]-C_c[0])),1)}mm** vers la **{Sens_x}** par rapport à la traverse basse.
        """)

    
    st.divider()
    col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
    with col_pdf2:
        st.subheader("📄 Rapport PDF Complet")
        st.write("Téléchargez un rapport PDF contenant les détails et visualisations de toutes les lignes")
        
        # Bouton pour générer le PDF
        if st.button("🔄 Générer le Rapport PDF", use_container_width=True, type="primary"):
            with st.spinner("Génération du rapport PDF en cours..."):
                try:
                    pdf_buffer = generer_rapport_pdf(df_affichage, graphs)
                    st.session_state['pdf_buffer'] = pdf_buffer
                    st.session_state['pdf_generated'] = True
                    st.success("✅ Rapport PDF généré avec succès !")
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération : {str(e)}")
        
        # Bouton de téléchargement si le PDF a été généré
        if st.session_state.get('pdf_generated', False):
            st.download_button(
                label="⬇️ Télécharger le Rapport PDF",
                data=st.session_state['pdf_buffer'],
                file_name="rapport_quadrilateres.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_button"
            )
            st.info(f"📋 Le rapport contient {len(df_affichage) + 1} pages")

else:
    st.info("📥 Importez un fichier Excel pour commencer l’analyse.")
        # Footer
st.caption("Application développée avec Streamlit et Shapely")
