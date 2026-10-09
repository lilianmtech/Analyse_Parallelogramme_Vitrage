import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl as xl
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle
from matplotlib.colors import to_rgba
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import sympy as sy
from math import *
from numpy import *
from shapely import *
from scipy.optimize import minimize_scalar
import builtins
import io
import requests
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import hashlib

class Vitrage:
    def __init__(self,cadre_0,cadre_def,Gamme,pf,raico = None,calage_lateral='Sans'):
        self.Gamme = Gamme
        self.A = cadre_def['A']
        self.B = cadre_def['B']
        self.C = cadre_def['C']
        self.D = cadre_def['D']
        self.A0 = cadre_0['A']
        self.B0 = cadre_0['B']
        self.C0 = cadre_0['C']
        self.D0 = cadre_0['D']

        self.calage_lateral = calage_lateral
        self.pf = pf
        self.raico = raico

        self.cale_bas = 100
        self.cale_laterale = 40

    def distance_signee(self, P, S1, S2, ref, inverser=False):
        """
        Distance perpendiculaire signée du point P (2D) à la droite (S1,S2).

        Positive si P est du même côté de la droite que le point de
        référence `ref` (le côté "intérieur" du cadre) ; négative sinon.
        `inverser=True` retourne le signe opposé (utile pour une borne
        intérieure, où le point conforme doit être du côté OPPOSÉ à `ref`).

        Remplace une simple différence de coordonnée X ou Y : celle-ci
        n'est une approximation valable de l'écart réel que si le bord
        contrôlé est proche de la verticale/horizontale. Pour un cadre
        très incliné (ex. lanterneau), l'écart de coordonnée peut avoir
        un signe différent de l'écart perpendiculaire réel.
        """
        S1 = array(S1[:2], dtype=float)
        S2 = array(S2[:2], dtype=float)
        P = array(P[:2], dtype=float)
        ref = array(ref[:2], dtype=float)

        d = S2 - S1
        n = array([-d[1], d[0]])
        n = n / linalg.norm(n)

        signe_ref = 1 if dot(ref - S1, n) >= 0 else -1
        distance = dot(P - S1, n) * signe_ref

        return -distance if inverser else distance

    def distance_point_plan(self, P1, P2, P3, Ptest):
            #Vecteur normal
        a = sy.Symbol('a')
        b = 1
        c = sy.Symbol('c')
        
        #Coeffiecient directeur de AB
        x1 = P2[0]-P1[0]
        y1 = P2[1]-P1[1]
        z1 = P2[2]-P1[2]
        
        #Coeffiecient directeur de AD
        x2 = P3[0]-P1[0]
        y2 = P3[1]-P1[1]
        z2 = P3[2]-P1[2]
        
        #Détermination des coef du vecteur normal
        eq1 = sy.Eq(a*x1 + b*y1 + c*z1 , 0)
        eq2 = sy.Eq(a*x2 + b*y2 + c*z2 , 0)
        sol1 = sy.solve((eq1,eq2),(a,c))
        
        if sol1 == []:
            distance = 0
        else:
            a=float(sol1[a])
            c=float(sol1[c])
            
            #Détermination des coef de l'équation de plan
            d = -a*P1[0]-b*P1[1]-c*P1[2]
            
            #Distance entre point et plan
            distance = abs(a*Ptest[0]+b*Ptest[1]+c*Ptest[2]+d)/sqrt(a**2+b**2+c**2)

        return distance

    def DiffHorsPlan(self):
        distances = {}
        distances["A"] = self.distance_point_plan(self.C, self.B, self.D, self.A)
        distances["B"] = self.distance_point_plan(self.D, self.A, self.C, self.B)
        distances["C"] = self.distance_point_plan(self.A, self.B, self.D, self.C)
        distances["D"] = self.distance_point_plan(self.B, self.A, self.C, self.D)
        print(distances)
        return builtins.max(distances.values())
    
    def inner_rect_with_offsets(self,quad, offset_top, offset_bottom, offset_lr):
        """
        quad : Polygon à 4 sommets
        offset_top : marge intérieure en haut
        offset_bottom : marge intérieure en bas
        offset_lr : marge intérieure gauche/droite (même valeur)
        """

        # Enveloppe du quadrilatère
        minx, miny, maxx, maxy = quad.bounds
        width  = maxx - minx
        height = maxy - miny

        # Centre du quadrilatère
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        # Facteurs d'échelle nécessaires
        sx = (width - 2 * offset_lr) / width
        sy = (height - offset_top - offset_bottom) / height

        # Étape 1 : réduction (scale)
        inner = affinity.scale(quad, xfact=sx, yfact=sy, origin=(cx, cy))

        # Étape 2 : correction verticale
        translate_y = (offset_bottom - offset_top) / 2
        inner = affinity.translate(inner, yoff=translate_y)

        return inner


    def extend_line(self,line, factor=1000):
        """
        Étend une ligne dans les deux directions.
        factor: facteur d'extension (1000 = très long)
        """
        # Obtenir les coordonnées du segment
        coords = list(line.coords)
        start, end = coords[0], coords[-1]
    
        # Calculer le vecteur directionnel
        dx = end[0] - start[0]
        dy = end[1] - start[1]
    
        # Étendre le segment dans les deux directions
        new_start = (start[0] - dx * factor, start[1] - dy * factor)
        new_end = (end[0] + dx * factor, end[1] + dy * factor)
    
        return LineString([new_start, new_end])

    def find_intersection_point(self,segment1, segment2, extend_if_needed=True):
        """
        Trouve le point d'intersection entre deux segments.
        Si pas d'intersection, étend les segments pour trouver l'intersection.
        """
        # Vérifier si les segments se croisent déjà
        if segment1.intersects(segment2):
            intersection = segment1.intersection(segment2)
            if isinstance(intersection, Point):
                return intersection
    
        # Si pas d'intersection et extension autorisée
        if extend_if_needed:
            # Étendre les deux segments
            extended1 = self.extend_line(segment1)
            extended2 = self.extend_line(segment2)
        
            # Vérifier l'intersection des segments étendus
            if extended1.intersects(extended2):
                intersection = extended1.intersection(extended2)
                if isinstance(intersection, Point):
                    return intersection
    
        return None, False

    def inner_quad_with_offsets(self,cadre, offset_top=0.0, offset_bottom=0.0, offset_lr=0.0):
        A0,B0,C0,D0,ini =list(cadre.exterior.coords)
        A_B=offset_curve(LineString([A0,B0]), -offset_top)
        D_A =offset_curve(LineString([D0,A0]), -offset_lr)
        C_B=offset_curve(LineString([C0,B0]), offset_lr)
        D_C=offset_curve(LineString([D0,C0]), offset_bottom)
    
        A = self.find_intersection_point(A_B,D_A)
        B = self.find_intersection_point(A_B,C_B)
        C = self.find_intersection_point(C_B,D_C)
        D = self.find_intersection_point(D_C,D_A)
    
        return Polygon([A,B,C,D])

    def distance_angle(self,angle, vitrage, cote, origine, support):
        rotated = affinity.rotate(vitrage, angle, origin=origine)
        A_V, B_V, C_V, D_V, ini = list(rotated.exterior.coords)
        if cote == 'gauche' :
            line_vit = LineString([A_V, D_V])
        elif cote == 'droite' :
            line_vit = LineString([B_V, C_V])
        return line_vit.distance(support)

    def Dechaussement(self):
        
        #--------------------Defintion des bornes--------------------
        
        'Mise en place d''un dictionnaire pour les paramétres'
        l = ('Pos_vit_lat','Pos_vit_haut','Tolerance epine','Tolerance traverse','Calage lateral')
        Raico={}
        
        if self.calage_lateral == 'Sans':
            Tm1 = self.raico[0]
            Tm2 = self.raico[1]
            Tt1 = self.raico[2]
            Tt2 = self.raico[3]

            pos_vit_lat = (self.Gamme/2-(Tm1+self.pf+Tm2))/2+Tm2
            pos_vit_haut = (self.Gamme/2-(Tt1+self.pf+Tt2))/2+Tt2
            jeu_borne_lat = (self.Gamme/2-(Tm1+self.pf+Tm2))/2
            jeu_borne_haut = (self.Gamme/2-(Tt1+self.pf+Tt2))/2

            donnees = (pos_vit_lat, pos_vit_haut, jeu_borne_lat, jeu_borne_haut, 0)

        elif self.calage_lateral == 'Avec':
            Tm = self.raico[0]
            Ca = self.raico[1]
            Jc = self.raico[2]
            Tt1 = self.raico[3]
            Tt2 = self.raico[4]
            Jv = self.raico[5]
        
            pos_vit_lat = Ca+Jc+Tm
            pos_vit_haut = self.Gamme/2-(Tt2+self.pf+Tt1+Jv)+Tt2
            jeu_borne_lat = self.Gamme/2 - (Ca+Jc+self.pf+Tm*2)
            jeu_borne_haut = self.Gamme/2-(Tt1+self.pf+Tt2+Jv)

            donnees = (pos_vit_lat, pos_vit_haut, jeu_borne_lat, jeu_borne_haut, Ca+Tm)

        for i in range(0,len(l)):
            Raico[l[i]] = donnees[i]
        
        #-----------Definition des cales---------

        longueur_traverse_basse = sqrt((self.D[0] - self.C[0])**2 + (self.D[1] - self.C[1])**2)

        gauche_0 = LineString([(self.D0[0],self.D0[1]),(self.C0[0],self.C0[1])]).interpolate(self.cale_bas)
        gauche = LineString([(self.D[0],self.D[1]),(self.C[0],self.C[1])]).interpolate(self.cale_bas)
        droite_0 = LineString([(self.C0[0],self.C0[1]),(self.D0[0],self.D0[1])]).interpolate(self.cale_bas)
        droite = LineString([(self.C[0],self.C[1]),(self.D[0],self.D[1])]).interpolate(self.cale_bas)
        T_gauche = gauche_0.y-gauche.y
        T_droite = droite_0.y-droite.y
        
        traverse_basse_gauche = LineString([(self.D[0],self.D[1]),(self.C[0],self.C[1])])
        TG = traverse_basse_gauche.parallel_offset(13.0,'left',join_style=2)
        TGS = traverse_basse_gauche.interpolate(self.cale_bas)
        Distance_cale_gauche=TG.project(TGS)
        Support_cale_gauche=TG.interpolate(Distance_cale_gauche)

        traverse_basse_droite = LineString([(self.C[0],self.C[1]),(self.D[0],self.D[1])])
        TD = traverse_basse_droite.parallel_offset(13.0,'right',join_style=2)
        TDS = traverse_basse_droite.interpolate(self.cale_bas)
        Distance_cale_droite=TD.project(TDS)
        Support_cale_droite=TD.interpolate(Distance_cale_droite)

        support_bas = {
    "T_gauche": T_gauche,
    "gauche":   Support_cale_gauche,
    "T_droite": T_droite,
    "droite":   Support_cale_droite
}

        cale_lateral_gauche = LineString([(self.A[0],self.A[1]),(self.D[0],self.D[1])])
        BG = cale_lateral_gauche.parallel_offset(Raico['Calage lateral'],'left',join_style=2)
        BGS = cale_lateral_gauche.interpolate(self.cale_laterale)
        Distance_support_gauche=BG.project(BGS)
        Support_lateral_gauche=BG.interpolate(Distance_support_gauche)

        cale_lateral_droite = LineString([(self.B[0],self.B[1]),(self.C[0],self.C[1])])
        BD = cale_lateral_droite.parallel_offset(Raico['Calage lateral'],'right',join_style=2)
        BDS = cale_lateral_droite.interpolate(self.cale_laterale)
        Distance_support_droite=BD.project(BDS)
        Support_lateral_droite=BD.interpolate(Distance_support_droite)

        support_laterale = {
            "gauche":   Support_lateral_gauche,
            "droite":   Support_lateral_droite
        }


        #-----------Definition du vitrage---------

        cadre_0 = Polygon([
        list(self.A0[0:2]),
        list(self.B0[0:2]),
        list(self.C0[0:2]),
        list(self.D0[0:2])
    ])  

        Vitrage = self.inner_quad_with_offsets( cadre_0, offset_top=Raico['Pos_vit_haut'], offset_bottom=13.0,offset_lr=Raico['Pos_vit_lat'])
              
        
        #-----------Mise en mouvement---------           
        if self.C0[1] >= self.C[1]:
            #longueur_traverse_basse=hypoténuse
            cote_support_bas = 'gauche'
            signe_rotation = -1
            #Angle positif → rotation horaire (vers la droite)
        else :
            cote_support_bas = 'droite'
            signe_rotation = 1
            #Angle négatif → rotation anti‑horaire (vers la gauche)
        
        Angle=degrees(asin((abs(self.D[1]-self.C[1]))/longueur_traverse_basse))

        #Translation du vitrage
        Vitrage_T = affinity.translate(Vitrage,xoff=0.0, yoff=-support_bas['T_'+cote_support_bas], zoff=0.0)
        #Rotation du vitrage'
        Vitrage_T_R = affinity.rotate(Vitrage_T, signe_rotation*Angle, origin=(support_bas[cote_support_bas]))

        contact_cale = 'Non'

        if self.calage_lateral == 'Avec' :
            #On vérifie si conflit avec les cales :
            if Vitrage_T_R.distance(support_laterale['droite'])==0:
                func = lambda a: self.distance_angle(a, Vitrage_T, 'droite',support_bas[cote_support_bas], support_laterale['droite'])
                res = minimize_scalar(func, bounds=(-10, 10), method='bounded')
                Vitrage_T_R = affinity.rotate(Vitrage_T, res.x, origin=support_bas[cote_support_bas])
                contact_cale = 'droite'

            elif Vitrage_T_R.distance(support_laterale['gauche'])==0:
                func = lambda a: self.distance_angle(a, Vitrage_T, 'gauche',support_bas[cote_support_bas], support_laterale['gauche'])
                res = minimize_scalar(func, bounds=(-10, 10), method='bounded')
                Vitrage_T_R = affinity.rotate(Vitrage_T, res.x, origin=support_bas[cote_support_bas])
                contact_cale = 'gauche'
    
        #-----------Vérifications des limites---------     
        
        A_V, B_V, C_V, D_V, ini = list(Vitrage_T_R.exterior.coords)

        Cadre = Polygon([
            list(self.A[0:2]),
            list(self.B[0:2]),
            list(self.C[0:2]),
            list(self.D[0:2])
        ])

        # Cadre de référence (non déformé) : nécessaire pour exprimer les
        # déplacements de traverse/montant en relatif plutôt qu'en absolu.
        Cadre_0 = Polygon([
            list(self.A0[0:2]),
            list(self.B0[0:2]),
            list(self.C0[0:2]),
            list(self.D0[0:2])
        ])

        if self.calage_lateral == 'Avec':
            Borne_ext = self.inner_quad_with_offsets( Cadre, offset_top=Raico['Pos_vit_haut']-Raico['Tolerance traverse'], offset_bottom=13.0,offset_lr=Raico['Calage lateral'])
            Borne_int = self.inner_quad_with_offsets( Cadre, offset_top=Raico['Pos_vit_haut']+Jv, offset_bottom=13.0,offset_lr=Raico['Pos_vit_lat']+Raico['Tolerance epine'])
        else :
            Borne_ext = self.inner_quad_with_offsets( Cadre, offset_top=Raico['Pos_vit_haut']-Raico['Tolerance traverse'], offset_bottom=13.0,offset_lr=Raico['Pos_vit_lat']-Raico['Tolerance epine'])
            Borne_int = self.inner_quad_with_offsets( Cadre, offset_top=Raico['Pos_vit_haut']+Raico['Tolerance traverse'], offset_bottom=13.0,offset_lr=Raico['Pos_vit_lat']+Raico['Tolerance epine'])


        A_BE, B_BE, C_BE, D_BE, ini = list(Borne_ext.exterior.coords)
        A_BI, B_BI, C_BI, D_BI, ini = list(Borne_int.exterior.coords)

        # Point de référence "intérieur" du cadre, pour orienter les distances signées
        centre = Cadre.centroid.coords[0]

        Ctrl_gauche_BE = self.distance_signee(A_V, D_BE, A_BE, centre)
        Ctrl_gauche_BI = self.distance_signee(A_V, D_BI, A_BI, centre, inverser=True)
        Ctrl_droite_BE = self.distance_signee(B_V, B_BE, C_BE, centre)
        Ctrl_droite_BI = self.distance_signee(B_V, B_BI, C_BI, centre, inverser=True)
        Ctrl_haut_BE = builtins.min(
            self.distance_signee(A_V, A_BE, B_BE, centre),
            self.distance_signee(B_V, A_BE, B_BE, centre),
        )
        Ctrl_haut_BI = builtins.min(
            self.distance_signee(A_V, A_BI, B_BI, centre, inverser=True),
            self.distance_signee(B_V, A_BI, B_BI, centre, inverser=True),
        )
        
        if self.calage_lateral == 'Avec' :
            if contact_cale == 'droite':
                Ctrl_droite_BE = 0

            elif contact_cale == 'gauche':
                Ctrl_gauche_BE = 0

        Vitrage_T_R = Polygon(Vitrage_T_R)
        
        data = [round(builtins.min(Ctrl_gauche_BE,Ctrl_gauche_BI),1),round(builtins.min(Ctrl_droite_BE,Ctrl_droite_BI),1),round(builtins.min(Ctrl_haut_BE,Ctrl_haut_BI),1)]
        graph = [Cadre, Borne_ext, Vitrage_T_R, Borne_int, Cadre_0]

        return data, graph

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

def _base_repere_local(A, B, C, D):
    """
    Construit la base orthonormée du repère local défini par le
    quadrilatère ABCD (voir docstring de `repere_local`).

    Retour :
        D, x_axis, y_axis, z_axis (np.array), tous exprimés dans le
        repère global.
    """
    A, B, C, D = (array(P, dtype=float) for P in (A, B, C, D))

    # Axe x : direction D -> C
    x_axis = C - D
    x_axis = x_axis / linalg.norm(x_axis)

    # Axe y : direction D -> A, orthogonalisée (Gram-Schmidt) par rapport à x
    da = A - D
    y_axis = da - dot(da, x_axis) * x_axis
    norme_y = linalg.norm(y_axis)
    if norme_y == 0:
        raise ValueError("A, C et D sont alignés : impossible de définir un axe y.")
    y_axis = y_axis / norme_y

    # Axe z : normale au plan (x, y)
    z_axis = cross(x_axis, y_axis)

    return D, x_axis, y_axis, z_axis

def repere_local(A, B, C, D):
    """
    Définit un repère local orthonormé à partir d'un quadrilatère ABCD
    exprimé dans le repère global (coordonnées 3D).

    Convention (vue de face du cadre) :
        A : coin haut-gauche      B : coin haut-droit
        D : coin bas-gauche       C : coin bas-droit

    Construction du repère local :
        - Origine : D -> (0, 0, 0)
        - Axe x   : dirigé de D vers C (bas du cadre)
        - Axe y   : direction D -> A, orthogonalisée (Gram-Schmidt)
                    par rapport à l'axe x
        - Axe z   : produit vectoriel x ^ y (normale au plan moyen)

    Si A, B, C, D sont rigoureusement coplanaires, les coordonnées locales
    z de chaque point valent 0. Si le quadrilatère est légèrement gauche
    (cadre déformé), z traduit l'écart hors-plan de chaque point par
    rapport au plan de référence (D, x, y).

    Paramètres
        A, B, C, D : tuple/list/array (x, y, z) - coordonnées globales

    Retour
        dict :
            'origine' : D en coordonnées globales (np.array)
            'base'    : (x_axis, y_axis, z_axis) exprimés dans le repère global
            'locales' : {'A':.., 'B':.., 'C':.., 'D':..} coordonnées locales (px, py, pz)
    """
    D_, x_axis, y_axis, z_axis = _base_repere_local(A, B, C, D)

    def vers_local(P):
        v = array(P, dtype=float) - D_
        return array([dot(v, x_axis), dot(v, y_axis), dot(v, z_axis)])

    locales = {label: vers_local(P) for label, P in zip("ABCD", (A, B, C, D))}

    return {
        'origine': D_,
        'base': (x_axis, y_axis, z_axis),
        'locales': locales,
    }

def local_coordinates(A, B, C, D, P):
    """
    Calcule les coordonnées locales du point P dans le repère local
    orthonormé défini par le quadrilatère ABCD (voir `repere_local`).

    Paramètres :
        A, B, C, D, P : tuples (x,y,z)

    Retour :
        Coordonnées locales de P (px, py, pz)
    """
    D_, x_axis, y_axis, z_axis = _base_repere_local(A, B, C, D)
    v = array(P, dtype=float) - D_

    return array([dot(v, x_axis), dot(v, y_axis), dot(v, z_axis)]).tolist()

#---------------------------Chargement des vitrages---------------------------"
def Dechaussement_vitrage(uploaded_file, option_calage, Raico, Typo):
    Feuille_coord = 'Coord_Points'
    Coord,Depl = CoordPoint(uploaded_file,Feuille_coord)
    
    document =xl.load_workbook(uploaded_file,data_only=True)
    sheet = document['Cadres']
    max_row = sheet.max_row
    datas = {
        'ID vitrage': [],
        'Gamme': [],
        'Demi-périmètre (m)': [],
        'Pf (mm)': [],
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
        
        cadre_0 = {}
        cadre_def = {}
        

        if Coord[labels["A"]][2] != 0 or Coord[labels["B"]][2] != 0 or Coord[labels["C"]][2] != 0 or Coord[labels["D"]][2] != 0 :
            Depl_rep = {}
            for  key, label in labels.items():
                cadre_0[key] = local_coordinates(Coord[labels["A"]], Coord[labels["B"]], Coord[labels["C"]], Coord[labels["D"]], Coord[label])

            for  key, label in labels.items():
                x = Depl[label][0] + Coord[labels["D"]][0]
                y = Depl[label][1] + Coord[labels["D"]][1]
                z = Depl[label][2] + Coord[labels["D"]][2]
                Depl_rep[key] = local_coordinates(Coord[labels["A"]], Coord[labels["B"]], Coord[labels["C"]], Coord[labels["D"]], [x,y,z])

            for key, label in labels.items():
                x = cadre_0[key][0] + (Depl_rep[key][0] - Depl_rep["D"][0])
                if key == "D":  # cas particulier pour D
                    x = 0

                if key == "A":
                    y = cadre_0[key][1]
                elif key == "B":
                    y = cadre_0[key][1] + (Depl_rep["C"][1] - Depl_rep["D"][1])
                elif key == "C":
                    y = cadre_0[key][1] + (Depl_rep["C"][1] - Depl_rep["D"][1])
                elif key == "D":
                    y = 0

                z = cadre_0[key][2] + (Depl_rep[key][2] - Depl_rep["D"][2])

                cadre_def[key] = [x, y, z]
        
        else :
            for key, label in labels.items():
                x = abs(Coord[label][0] - Coord[labels["D"]][0]) + (Depl[label][0] - Depl[labels["D"]][0])
                if key == "D":  # cas particulier pour D
                    x = 0

                if key == "A":
                    y = abs(Coord[label][1] - Coord[labels["D"]][1])
                elif key == "B":
                    y = abs(Coord[label][1] - Coord[labels["D"]][1]) + (Depl[labels["C"]][1] - Depl[labels["D"]][1])
                elif key == "C":
                    y = abs(Coord[label][1] - Coord[labels["D"]][1]) + (Depl[labels["C"]][1] - Depl[labels["D"]][1])
                elif key == "D":
                    y = 0

                z = (Coord[label][2] - Coord[labels["D"]][2]) + (Depl[label][2] - Depl[labels["D"]][2])

                cadre_def[key] = [x, y, z]

                x_0 = abs (Coord[label][0] - Coord[labels["D"]][0])
                y_0 = abs (Coord[label][1] - Coord[labels["D"]][1])
                z_0 = Coord[label][2] - Coord[labels["D"]][2]

                cadre_0[key] = [x_0, y_0, z_0]
        
        H =sqrt((cadre_0['A'][0] - cadre_0['D'][0])**2 + (cadre_0['A'][1] - cadre_0['D'][1])**2)
        L = sqrt((cadre_0['A'][0] - cadre_0['B'][0])**2 + (cadre_0['A'][1] - cadre_0['B'][1])**2)
        demiperi = L/1000 + H/1000
        if Typo == 'Façade':
            if demiperi > 7.0 :
                pf = 12.0
            elif  5.0 < demiperi <= 7.0 :
                pf = 9.0
            elif demiperi <= 5.0 :
                pf = 6.0
        elif Typo == 'Verrière':
            if H <= 1000 or L <= 1000 :
                pf = 8.0
            else :
                pf = 10.0
        V = Vitrage(cadre_0,cadre_def,Gamme,pf,raico=Raico,calage_lateral=option_calage)
        data, graph = V.Dechaussement()

        datas['ID vitrage'].append(str(labels["A"])+' / '+str(labels["B"])+' / '+str(labels["C"])+' / '+str(labels["D"]))
        datas['Gamme'].append(Gamme)
        datas['Demi-périmètre (m)'].append(round(demiperi,1))
        datas['Pf (mm)'].append(round(pf,1))
        datas['Ecart minimal // bornes gauche (mm)'].append(data[0])
        datas['Ecart minimal // bornes droite (mm)'].append(data[1])
        datas['Ecart minimal // bornes hautes (mm)'].append(data[2])

        graphs['ID vitrage'].append(str(labels["A"])+' / '+str(labels["B"])+' / '+str(labels["C"])+' / '+str(labels["D"]))
        graphs['Graph'].append(graph)
        
    return datas, graphs

def Gauchissement_vitrage(uploaded_file):
    Feuille_coord = 'Coord_Points'
    Coord,Depl = CoordPoint(uploaded_file,Feuille_coord)
    
    document =xl.load_workbook(uploaded_file,data_only=True)
    sheet = document['Cadres']
    max_row = sheet.max_row
    datas = {
        'ID vitrage': [],
        'Gauchissement (mm)': [],
        'L (mm)' : [],
        'Diag (mm)' : [],
        'Critère cahier 3574v2 (mm)*': [],
        'Critère DTU39-P4 (mm)**': [],
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
        
        cadre_0 = {}
        cadre_def = {}
        

        if Coord[labels["A"]][2] != 0 or Coord[labels["B"]][2] != 0 or Coord[labels["C"]][2] != 0 or Coord[labels["D"]][2] != 0 :
            Depl_rep = {}
            for  key, label in labels.items():
                cadre_0[key] = local_coordinates(Coord[labels["A"]], Coord[labels["B"]], Coord[labels["C"]], Coord[labels["D"]], Coord[label])

            for  key, label in labels.items():
                x = Depl[label][0] + Coord[labels["D"]][0]
                y = Depl[label][1] + Coord[labels["D"]][1]
                z = Depl[label][2] + Coord[labels["D"]][2]
                Depl_rep[key] = local_coordinates(Coord[labels["A"]], Coord[labels["B"]], Coord[labels["C"]], Coord[labels["D"]], [x,y,z])

            for key, label in labels.items():
                x = cadre_0[key][0] + (Depl_rep[key][0] - Depl_rep["D"][0])
                if key == "D":  # cas particulier pour D
                    x = 0

                if key == "A":
                    y = cadre_0[key][1]
                elif key == "B":
                    y = cadre_0[key][1] + (Depl_rep["C"][1] - Depl_rep["D"][1])
                elif key == "C":
                    y = cadre_0[key][1] + (Depl_rep["C"][1] - Depl_rep["D"][1])
                elif key == "D":
                    y = 0

                z = cadre_0[key][2] + (Depl_rep[key][2] - Depl_rep["D"][2])

                cadre_def[key] = [x, y, z]
        
        else :
            for key, label in labels.items():
                x = abs(Coord[label][0] - Coord[labels["D"]][0]) + (Depl[label][0] - Depl[labels["D"]][0])
                if key == "D":  # cas particulier pour D
                    x = 0

                if key == "A":
                    y = abs(Coord[label][1] - Coord[labels["D"]][1])
                elif key == "B":
                    y = abs(Coord[label][1] - Coord[labels["D"]][1]) + (Depl[labels["C"]][1] - Depl[labels["D"]][1])
                elif key == "C":
                    y = abs(Coord[label][1] - Coord[labels["D"]][1]) + (Depl[labels["C"]][1] - Depl[labels["D"]][1])
                elif key == "D":
                    y = 0

                z = (Coord[label][2] - Coord[labels["D"]][2]) + (Depl[label][2] - Depl[labels["D"]][2])

                cadre_def[key] = [x, y, z]

                x_0 = abs (Coord[label][0] - Coord[labels["D"]][0])
                y_0 = abs (Coord[label][1] - Coord[labels["D"]][1])
                z_0 = Coord[label][2] - Coord[labels["D"]][2]

                cadre_0[key] = [x_0, y_0, z_0]

        H = sqrt((cadre_0['A'][0] - cadre_0['D'][0])**2 + (cadre_0['A'][1] - cadre_0['D'][1])**2)
        L = sqrt((cadre_0['A'][0] - cadre_0['B'][0])**2 + (cadre_0['A'][1] - cadre_0['B'][1])**2)
        diag1 = sqrt((cadre_0['A'][0] - cadre_0['C'][0])**2 + (cadre_0['A'][1] - cadre_0['C'][1])**2)
        diag2 = sqrt((cadre_0['B'][0] - cadre_0['D'][0])**2 + (cadre_0['B'][1] - cadre_0['D'][1])**2)
        pf = L/1000 + H/1000
        print(cadre_0,cadre_def)
        V = Vitrage(cadre_0,cadre_def,Gamme,pf)
        data = V.DiffHorsPlan()

        datas['ID vitrage'].append(str(labels["A"])+' / '+str(labels["B"])+' / '+str(labels["C"])+' / '+str(labels["D"]))
        datas['Gauchissement (mm)'].append(data)
        datas['L (mm)'].append(builtins.min(H,L))
        datas['Diag (mm)'].append(builtins.min(diag1,diag2))
        datas['Critère cahier 3574v2 (mm)*'].append(builtins.min(H,L)/75)
        datas['Critère DTU39-P4 (mm)**'].append(builtins.min(diag1,diag2)/150) 
    return datas

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-----------------------Charte graphique du rapport----------------------------

COULEUR_PRIMAIRE = "#006D8F"
COULEUR_PRIMAIRE_FONCEE = "#00485E"
COULEUR_CADRE = "#2B2B2B"
COULEUR_LIMITE = "#D64545"
COULEUR_VITRAGE = "#006D8F"
COULEUR_OK = "#1E7A34"
COULEUR_OK_FOND = "#E5F3E8"
COULEUR_KO = "#B3261E"
COULEUR_KO_FOND = "#FBE9E7"
COULEUR_LIGNE_PAIRE = "#F2F6F7"
COULEUR_GRILLE = "#D9D9D9"

#-----------------------visualiser les quadrilatères avec zoom----------------

def visualiser_quadrilateres(quadrilateres, titre_page=None):
    """Crée deux graphiques avec zoom adaptatif sur les coins supérieurs"""
    # Format A4 en paysage : 11.69 x 8.27 pouces
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))

    # Titre de la page si fourni
    if titre_page:
        fig.suptitle(titre_page, fontsize=16, fontweight='bold', color=COULEUR_PRIMAIRE_FONCEE, y=0.98)

    couleurs = [COULEUR_CADRE, COULEUR_LIMITE, COULEUR_VITRAGE, COULEUR_LIMITE]
    titre_graph =['Cadre','Limite','Vitrage','Limite']

    # Tracer les quadrilatères sur les deux axes - le vitrage est mis en évidence par un remplissage
    # (seuls les 4 premiers éléments sont destinés à être dessinés ; un éventuel
    # 5e élément est le cadre de référence, utilisé uniquement pour les calculs)
    for ax in [ax1, ax2]:
        for i, quad in enumerate(quadrilateres[:4]):
            x, y = quad.exterior.xy
            est_vitrage = (i == 2)
            patch = MplPolygon(list(zip(x, y)),
                             facecolor=to_rgba(couleurs[i], 0.12) if est_vitrage else 'none',
                             edgecolor=couleurs[i], linewidth=2.0 if est_vitrage else 1.2,
                             label=titre_graph[i])
            ax.add_patch(patch)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color=COULEUR_GRILLE)
        ax.legend(framealpha=0.9)
    
    # Zoom adaptatif sur le coin supérieur gauche
    quad1 = quadrilateres[2]
    x1, y1 = quad1.exterior.xy
    
    # Coin supérieur gauche = sommet A du vitrage (convention du repère local),
    # pas une approximation par coordonnées : celle-ci se trompait de coin sur
    # un cadre très incliné (deux sommets peuvent être simultanément "x min, y max").
    points1 = list(zip(x1, y1))
    coin_haut_gauche = points1[0]
    
    # Calculer la taille de la zone de zoom basée sur la taille du quadrilatère
    largeur_quad1 = builtins.max(x1) - builtins.min(x1)
    hauteur_quad1 = builtins.max(y1) - builtins.min(y1)
    marge = builtins.max(largeur_quad1, hauteur_quad1) * 0.01  # 1% de marge
    
    ax1.set_xlim(coin_haut_gauche[0] - marge, coin_haut_gauche[0] + marge)
    ax1.set_ylim(coin_haut_gauche[1] - marge, coin_haut_gauche[1] + marge)
    ax1.set_title("Zoom - Coin Supérieur Gauche", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    
    # Zoom adaptatif sur le coin supérieur droit
    quad4 = quadrilateres[2]
    x4, y4 = quad4.exterior.xy
    
    # Coin supérieur droit = sommet B du vitrage (convention du repère local)
    points4 = list(zip(x4, y4))
    coin_haut_droit = points4[1]
    
    # Calculer la taille de la zone de zoom basée sur la taille du quadrilatère
    largeur_quad4 = builtins.max(x4) - builtins.min(x4)
    hauteur_quad4 = builtins.max(y4) - builtins.min(y4)
    marge = builtins.max(largeur_quad4, hauteur_quad4) * 0.01  # 1% de marge
    
    ax2.set_xlim(coin_haut_droit[0] - marge, coin_haut_droit[0] + marge)
    ax2.set_ylim(coin_haut_droit[1] - marge, coin_haut_droit[1] + marge)
    ax2.set_title("Zoom - Coin Supérieur Droit", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    
    plt.tight_layout()
    return fig

#----------------------Edition du rapport pdf FORMAT A4 PAYSAGE---------------

def texte_deplacement_relatif(cadre_def, cadre_ref):
    """
    Décrit le déplacement de la traverse haute (A-B) par rapport à la traverse
    basse (D-C), et du montant droit par rapport au montant gauche.

    Le déplacement de la traverse est exprimé en RELATIF par rapport à la
    géométrie de référence (non déformée), et non en absolu : une comparaison
    absolue des coordonnées X de A et D donne un résultat dominé par la forme
    propre du cadre (un cadre très rampant, comme un lanterneau, peut avoir un
    grand écart X entre A et D même sans aucun déplacement réel).

    cadre_def : polygone Shapely à 4 sommets (A,B,C,D) — position actuelle
    cadre_ref : polygone Shapely à 4 sommets (A,B,C,D) — position de référence
    """
    A_c, B_c, C_c, D_c, _ = cadre_def.exterior.coords
    A_r, B_r, C_r, D_r, _ = cadre_ref.exterior.coords

    # Déplacement horizontal de chaque sommet par rapport à sa position de référence
    decalage_haut = ((A_c[0] - A_r[0]) + (B_c[0] - B_r[0])) / 2
    decalage_bas = ((D_c[0] - D_r[0]) + (C_c[0] - C_r[0])) / 2
    decalage_relatif = round(decalage_haut - decalage_bas, 1)
    Sens_x = 'droite' if decalage_relatif > 0 else 'gauche'

    # Le repère local est construit sur D-C de référence (D0.y = C0.y = 0),
    # donc D_c[1] et C_c[1] mesurent déjà un déplacement vertical relatif.
    descente_montant = round(abs(D_c[1] - C_c[1]), 1)
    if D_c[1] > C_c[1]:
        Sens_y, opp = 'droite', 'gauche'
    else:
        Sens_y, opp = 'gauche', 'droite'

    texte_montant = f"Le montant de {Sens_y} descend de {descente_montant}mm par rapport au montant de {opp}."
    texte_traverse = f"La traverse haute se déplace de {abs(decalage_relatif)}mm vers la {Sens_x} par rapport à la traverse basse."

    if descente_montant == 0 and abs(decalage_relatif) == 0:
        return "Aucun déplacement significatif détecté entre la traverse basse et la traverse haute."
    if descente_montant == 0:
        return texte_traverse
    if abs(decalage_relatif) == 0:
        return texte_montant
    return texte_montant + "\n" + texte_traverse

def creer_page_complete(ligne_data, quadrilateres, page_num=None, total_pages=None):
    """Crée une page A4 PAYSAGE avec les détails et les deux graphiques côte à côte"""
    # Format A4 en PAYSAGE : 11.69 x 8.27 pouces
    fig = plt.figure(figsize=(11.69, 8.27))

    # Récupérer les valeurs des écarts
    ecart_gauche = ligne_data['Ecart minimal // bornes gauche (mm)']
    ecart_droite = ligne_data['Ecart minimal // bornes droite (mm)']
    ecart_hautes = ligne_data['Ecart minimal // bornes hautes (mm)']
    conforme = (ecart_gauche >= 0) and (ecart_droite >= 0) and (ecart_hautes >= 0)

    # Créer une grille pour organiser le contenu en PAYSAGE
    # 5 lignes, 2 colonnes : [Bandeau titre] [Métriques] [espace] [Graph gauche | Graph droit] [Annotation]
    # La ligne "espace" garantit un vrai espace blanc entre le tableau et les titres des graphiques.
    gs = fig.add_gridspec(5, 2, height_ratios=[0.09, 0.20, 0.05, 0.53, 0.13], hspace=0.3, wspace=0.20,
                          left=0.06, right=0.97, top=0.94, bottom=0.06)

    # Section 0 : bandeau d'en-tête avec l'identifiant du vitrage et le statut de conformité
    ax_entete = fig.add_subplot(gs[0, :])
    ax_entete.axis('off')
    ax_entete.set_xlim(0, 1)
    ax_entete.set_ylim(0, 1)
    ax_entete.add_patch(MplRectangle((0, 0), 1, 1, transform=ax_entete.transAxes,
                                      facecolor=COULEUR_PRIMAIRE, edgecolor='none'))
    ax_entete.text(0.015, 0.5, f"Vitrage {ligne_data['ID vitrage']}", transform=ax_entete.transAxes,
                   ha='left', va='center', fontsize=14, fontweight='bold', color='white', family='Sans')
    statut_txt = "CONFORME" if conforme else "NON CONFORME"
    statut_couleur = COULEUR_OK if conforme else COULEUR_KO
    ax_entete.text(0.985, 0.5, statut_txt, transform=ax_entete.transAxes,
                   ha='right', va='center', fontsize=12, fontweight='bold', color='white', family='Sans',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=statut_couleur, edgecolor='none'))

    # Section 1: Tableau des métriques principales (sous le bandeau, sur toute la largeur)
    ax_metrics = fig.add_subplot(gs[1, :])
    ax_metrics.axis('tight')
    ax_metrics.axis('off')

    def ligne_ecart(libelle, valeur):
        statut = "✓ Conforme" if valeur >= 0 else "✗ Non conforme"
        return [libelle, f"{valeur:.1f}", 'mm', statut]

    metrics_data = [
        ['Paramètre', 'Valeur', 'Unité', 'Statut'],
        ['Gamme', f"{ligne_data['Gamme']}", '-', ''],
        ['Pf mini', f"{ligne_data['Pf (mm)']:.1f}", 'mm', ''],
        ligne_ecart('Écart minimal // bornes gauche', ecart_gauche),
        ligne_ecart('Écart minimal // bornes droite', ecart_droite),
        ligne_ecart('Écart minimal // bornes hautes', ecart_hautes),
    ]
    ecarts_par_ligne = {3: ecart_gauche, 4: ecart_droite, 5: ecart_hautes}

    table = ax_metrics.table(cellText=metrics_data, cellLoc='center', loc='center',
                            colWidths=[0.32, 0.10, 0.08, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style du tableau : en-tête teal, lignes zébrées, écarts colorés selon conformité
    for i in range(len(metrics_data)):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_edgecolor(COULEUR_GRILLE)
            if i == 0:  # En-tête
                cell.set_facecolor(COULEUR_PRIMAIRE)
                cell.set_text_props(weight='bold', color='white')
            elif i in ecarts_par_ligne and j in (1, 3):
                ok = ecarts_par_ligne[i] >= 0
                cell.set_facecolor(COULEUR_OK_FOND if ok else COULEUR_KO_FOND)
                cell.set_text_props(color=COULEUR_OK if ok else COULEUR_KO, weight='bold')
            else:
                cell.set_facecolor(COULEUR_LIGNE_PAIRE if i % 2 == 0 else 'white')

    couleurs = [COULEUR_CADRE, COULEUR_LIMITE, COULEUR_VITRAGE, COULEUR_LIMITE]
    titre_graph = ['Cadre','Limite','Vitrage','Limite']

    # Section 3: GRAPHIQUE GAUCHE - Zoom coin supérieur gauche
    ax1 = fig.add_subplot(gs[3, 0])

    for i, quad in enumerate(quadrilateres[:4]):
        x, y = quad.exterior.xy
        est_vitrage = (i == 2)
        patch = MplPolygon(list(zip(x, y)),
                         facecolor=to_rgba(couleurs[i], 0.12) if est_vitrage else 'none',
                         edgecolor=couleurs[i], linewidth=2.0 if est_vitrage else 1.3,
                         label=titre_graph[i])
        ax1.add_patch(patch)

    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, color=COULEUR_GRILLE, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Zoom adaptatif sur le coin supérieur gauche
    quad1 = quadrilateres[2]
    x1, y1 = quad1.exterior.xy
    # Coin supérieur gauche = sommet A du vitrage (convention du repère local)
    points1 = list(zip(x1, y1))
    coin_haut_gauche = points1[0]

    largeur_quad1 = builtins.max(x1) - builtins.min(x1)
    hauteur_quad1 = builtins.max(y1) - builtins.min(y1)
    marge = builtins.max(largeur_quad1, hauteur_quad1) * 0.01
    
    ax1.set_xlim(coin_haut_gauche[0] - marge, coin_haut_gauche[0] + marge)
    ax1.set_ylim(coin_haut_gauche[1] - marge, coin_haut_gauche[1] + marge)
    ax1.set_title("Coin Supérieur Gauche",
                  fontsize=11, fontweight='bold', color=COULEUR_PRIMAIRE_FONCEE, pad=10)
    ax1.set_xlabel("Coordonnée X (mm)", fontsize=9)
    ax1.set_ylabel("Coordonnée Y (mm)", fontsize=9)
    ax1.tick_params(labelsize=8)

    # Section 4: GRAPHIQUE DROIT - Zoom coin supérieur droit
    ax2 = fig.add_subplot(gs[3, 1])

    for i, quad in enumerate(quadrilateres[:4]):
        x, y = quad.exterior.xy
        est_vitrage = (i == 2)
        patch = MplPolygon(list(zip(x, y)),
                         facecolor=to_rgba(couleurs[i], 0.12) if est_vitrage else 'none',
                         edgecolor=couleurs[i], linewidth=2.0 if est_vitrage else 1.3,
                         label=titre_graph[i])
        ax2.add_patch(patch)

    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, color=COULEUR_GRILLE, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Zoom adaptatif sur le coin supérieur droit
    quad4 = quadrilateres[2]
    x4, y4 = quad4.exterior.xy
    # Coin supérieur droit = sommet B du vitrage (convention du repère local)
    points4 = list(zip(x4, y4))
    coin_haut_droit = points4[1]

    largeur_quad4 = builtins.max(x4) - builtins.min(x4)
    hauteur_quad4 = builtins.max(y4) - builtins.min(y4)
    marge = builtins.max(largeur_quad4, hauteur_quad4) * 0.01
    
    ax2.set_xlim(coin_haut_droit[0] - marge, coin_haut_droit[0] + marge)
    ax2.set_ylim(coin_haut_droit[1] - marge, coin_haut_droit[1] + marge)
    ax2.set_title("Coin Supérieur Droit",
                  fontsize=11, fontweight='bold', color=COULEUR_PRIMAIRE_FONCEE, pad=10)
    ax2.set_xlabel("Coordonnée X (mm)", fontsize=9)
    ax2.set_ylabel("Coordonnée Y (mm)", fontsize=9)
    ax2.tick_params(labelsize=8)

    # Section 2: Annotation (sur toute la largeur)
    ax_annotation = fig.add_subplot(gs[4, :])
    ax_annotation.axis('off')

    cadre = quadrilateres[0]
    cadre_ref = quadrilateres[4]
    annotation_text = texte_deplacement_relatif(cadre, cadre_ref)

    ax_annotation.text(0.5, 0.65, annotation_text,
                      ha='center', va='center',
                      fontsize=9, family='Sans',
                      bbox=dict(boxstyle='square,pad=0.3', facecolor="#EDF4F5",
                               edgecolor=COULEUR_PRIMAIRE, linewidth=1, alpha=0.9))

    # Pied de page : identification du document et pagination
    if page_num is not None and total_pages is not None:
        fig.text(0.5, 0.015,
                  f"MTECHBUILD — Analyse de la mise en parallélogramme des vitrages    |    Page {page_num}/{total_pages}",
                  ha='center', va='center', fontsize=7, color='#777777', family='Sans')

    return fig

def creer_page_parametres(Typo, choix, Raico, page_num=None, total_pages=None):
    """
    Page présentant les paramètres et tolérances retenus pour l'analyse,
    ainsi que les schémas de référence des bornes (montants et traverses).
    """
    fig = plt.figure(figsize=(11.69, 8.27))

    marge_gauche, marge_droite = 0.06, 0.97
    haut_page = 0.94
    bas_page = 0.16  # laisse la place au logo en filigrane, tamponné en bas à droite

    # Bandeau d'en-tête (même style que les pages vitrage)
    hauteur_bandeau = 0.09
    ax_entete = fig.add_axes([0, haut_page - hauteur_bandeau, 1, hauteur_bandeau])
    ax_entete.axis('off')
    ax_entete.set_xlim(0, 1)
    ax_entete.set_ylim(0, 1)
    ax_entete.add_patch(MplRectangle((0, 0), 1, 1, transform=ax_entete.transAxes,
                                      facecolor=COULEUR_PRIMAIRE, edgecolor='none'))
    ax_entete.text(0.015, 0.5, "Paramètres considérés", transform=ax_entete.transAxes,
                   ha='left', va='center', fontsize=14, fontweight='bold', color='white', family='Sans')

    # Construction des lignes du tableau (leur nombre dépend du calage latéral :
    # "Avec" compte 2 lignes de plus que "Sans")
    lignes = [['Paramètre', 'Valeur', 'Unité']]
    lignes.append(['Typologie', str(Typo), '-'])
    lignes.append(['Calage latéral', str(choix), '-'])
    if choix == 'Sans':
        Tm1, Tm2, Tt1, Tt2 = Raico
        lignes += [
            ['Tolérance Tm1', f"{Tm1:.2f}", 'mm'],
            ['Tolérance Tm2 + menuiserie', f"{Tm2:.2f}", 'mm'],
            ['Tolérance Tt1', f"{Tt1:.2f}", 'mm'],
            ['Tolérance Tt2 + menuiserie', f"{Tt2:.2f}", 'mm'],
        ]
    else:
        Tm, Ca, Jc, Tt1, Tt2, Jv = Raico
        lignes += [
            ['Tolérance Tm', f"{Tm:.2f}", 'mm'],
            ['Épaisseur cale C + menuiserie', f"{Ca:.2f}", 'mm'],
            ['Jeu entre vitrage et cale Jc', f"{Jc:.2f}", 'mm'],
            ['Tolérance Tt1', f"{Tt1:.2f}", 'mm'],
            ['Tolérance Tt2 + menuiserie', f"{Tt2:.2f}", 'mm'],
            ['Jeu entre borne basse et vitrage Jv', f"{Jv:.2f}", 'mm'],
        ]

    # Zone du tableau : hauteur calculée à partir du nombre réel de lignes (et non
    # une hauteur fixe), pour ne jamais empiéter sur la zone des schémas en
    # dessous — c'est ce qui provoquait le chevauchement avec le calage activé.
    hauteur_ligne = 0.032
    hauteur_tableau = len(lignes) * hauteur_ligne
    marge_sous_bandeau = 0.03
    y_haut_tableau = haut_page - hauteur_bandeau - marge_sous_bandeau
    y_bas_tableau = y_haut_tableau - hauteur_tableau

    ax_table = fig.add_axes([marge_gauche, y_bas_tableau, marge_droite - marge_gauche, hauteur_tableau])
    ax_table.axis('tight')
    ax_table.axis('off')

    table = ax_table.table(cellText=lignes, cellLoc='center', loc='center', colWidths=[0.36, 0.14, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for i in range(len(lignes)):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_edgecolor(COULEUR_GRILLE)
            if i == 0:
                cell.set_facecolor(COULEUR_PRIMAIRE)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(COULEUR_LIGNE_PAIRE if i % 2 == 0 else 'white')

    # Schémas de référence : bornes sur montants / bornes sur traverses.
    # Placés juste sous le tableau (quel que soit son nombre de lignes), et
    # descendant jusqu'à la marge basse fixe.
    url_schemas = 'https://github.com/lilianmtech/Analyse_Parallelogramme_Vitrage/blob/main/'
    nom_montant = "Borne_M_C.png" if choix == "Avec" else "Borne_M.png"
    nom_traverse = "Borne_T_C.png" if choix == "Avec" else "Borne_T.png"

    marge_avant_images = 0.04
    y_haut_images = y_bas_tableau - marge_avant_images
    hauteur_images = y_haut_images - bas_page
    largeur_images = (marge_droite - marge_gauche - 0.05) / 2

    ax_img1 = fig.add_axes([marge_gauche, bas_page, largeur_images, hauteur_images])
    ax_img2 = fig.add_axes([marge_droite - largeur_images, bas_page, largeur_images, hauteur_images])
    for ax, nom_fichier, legende in [
        (ax_img1, nom_montant, "Bornes sur montants"),
        (ax_img2, nom_traverse, "Bornes sur traverses"),
    ]:
        try:
            reponse = requests.get(url_schemas + nom_fichier + "?raw=true")
            img = mpimg.imread(io.BytesIO(reponse.content), format='png')
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image indisponible", ha='center', va='center', fontsize=10, color='gray')
        ax.axis('off')
        ax.set_title(legende, fontsize=12, fontweight='bold', color=COULEUR_PRIMAIRE_FONCEE, pad=8)

    # Pied de page : identification du document et pagination
    if page_num is not None and total_pages is not None:
        fig.text(0.5, 0.015,
                  f"MTECHBUILD — Analyse de la mise en parallélogramme des vitrages    |    Page {page_num}/{total_pages}",
                  ha='center', va='center', fontsize=7, color='#777777', family='Sans')

    return fig

# Fonction pour générer le rapport PDF complet
def generer_rapport_pdf(df, graphs, Typo, choix, Raico, titre_personnalise=None):
    """Génère un rapport PDF avec toutes les lignes au format A4 PAYSAGE

    titre_personnalise : texte optionnel affiché sur la page de garde, entre
    le titre principal et la ligne Gamme/Date (ex : nom du projet/chantier).
    """
    buffer = io.BytesIO()
    
    from datetime import datetime
    date_rapport = datetime.now().strftime("%d/%m/%Y")
    total_pages = 2 + len(df)  # page de garde + page paramètres + une page par vitrage

    with PdfPages(buffer) as pdf:
        # Page de titre - Format A4 PAYSAGE
        fig_titre = plt.figure(figsize=(11.69, 8.27))  # A4 paysage en pouces

        # Bandeau institutionnel (laissé vide, volontairement)
        fig_titre.add_artist(MplRectangle((0, 0.90), 1, 0.10, transform=fig_titre.transFigure,
                                           facecolor=COULEUR_PRIMAIRE, edgecolor='none'))

        # Titre principal
        fig_titre.text(0.5, 0.76, "ANALYSE DE LA MISE EN PARALLÉLOGRAMME\nDES VITRAGES",
                       ha='center', va='center',
                       fontsize=19, fontweight='bold', color=COULEUR_PRIMAIRE_FONCEE, family='Sans')

        # Titre personnalisé (optionnel, ex : nom du projet/chantier) : encadré plein
        # pour bien se détacher du fond blanc de la page.
        if titre_personnalise:
            boite_largeur, boite_hauteur = 0.6, 0.08
            boite_x = 0.5 - boite_largeur / 2
            boite_y = 0.58 - boite_hauteur / 2
            fig_titre.add_artist(MplRectangle((boite_x, boite_y), boite_largeur, boite_hauteur,
                                               transform=fig_titre.transFigure,
                                               facecolor=COULEUR_PRIMAIRE, edgecolor='none'))
            fig_titre.text(0.5, 0.58, titre_personnalise, ha='center', va='center',
                           fontsize=15, fontweight='bold', color='white', family='Sans')

        # Logo en bas à droite : taille calculée à partir d'une hauteur cible en
        # fraction de page (le ratio largeur/hauteur d'origine est conservé).
        url_logo = 'https://github.com/lilianmtech/Analyse_Parallelogramme_Vitrage/blob/main/logo-couleur.png?raw=true'
        response = requests.get(url_logo)
        img = mpimg.imread(io.BytesIO(response.content))
        orig_height, orig_width = img.shape[:2]

        fig_width_in, fig_height_in = fig_titre.get_size_inches()
        hauteur_cible_frac = 0.11  # 11% de la hauteur de la page
        img_height_frac = hauteur_cible_frac
        img_width_frac = img_height_frac * (orig_width / orig_height) * (fig_height_in / fig_width_in)

        right_margin = 0.03
        bottom_margin = 0.03
        left = 1 - right_margin - img_width_frac
        bottom = bottom_margin

        ax = fig_titre.add_axes([left, bottom, img_width_frac, img_height_frac])
        ax.imshow(img)
        ax.axis("off")

        # Résumé
        mask = (df[["Ecart minimal // bornes gauche (mm)",
                    "Ecart minimal // bornes droite (mm)",
                    "Ecart minimal // bornes hautes (mm)"]] < 0).any(axis=1)
        nb_non_conforme = mask.sum()
        nb_conforme = len(df) - nb_non_conforme

        Gammes = df["Gamme"].value_counts().index
        Gamme = " & ".join(map(str, Gammes))

        fig_titre.text(0.5, 0.42, f"Gamme Raico étudiée : {Gamme}    |    Date du rapport : {date_rapport}",
                       ha='center', va='center', fontsize=11, color='#444444', family='Sans')

        # Cartes indicateurs (synthèse chiffrée)
        kpi_y, kpi_h = 0.20, 0.15
        card_w, gap = 0.22, 0.035
        start_x = (1 - (3 * card_w + 2 * gap)) / 2
        kpis = [
            ("Vitrages étudiés", str(len(df)), COULEUR_PRIMAIRE),
            ("Conformes", str(nb_conforme), COULEUR_OK),
            ("Non conformes", str(nb_non_conforme), COULEUR_KO if nb_non_conforme > 0 else COULEUR_OK),
        ]
        for i, (label, value, couleur) in enumerate(kpis):
            x = start_x + i * (card_w + gap)
            fig_titre.add_artist(MplRectangle((x, kpi_y), card_w, kpi_h, transform=fig_titre.transFigure,
                                               facecolor='#F4F6F7', edgecolor=couleur, linewidth=1.6))
            fig_titre.text(x + card_w / 2, kpi_y + kpi_h * 0.62, value,
                           ha='center', va='center', fontsize=24, fontweight='bold', color=couleur, family='Sans')
            fig_titre.text(x + card_w / 2, kpi_y + kpi_h * 0.22, label,
                           ha='center', va='center', fontsize=9, color='#333333', family='Sans')

        # Pied de page
        fig_titre.text(0.97, 0.02, f"Page 1/{total_pages}",
                       ha='right', va='center', fontsize=7, color='#888888', family='Sans')

        pdf.savefig(fig_titre, bbox_inches='tight')
        plt.close(fig_titre)

        # Page des paramètres et tolérances retenus pour l'analyse
        fig_parametres = creer_page_parametres(Typo, choix, Raico, page_num=2, total_pages=total_pages)
        pdf.savefig(fig_parametres, bbox_inches='tight')
        plt.close(fig_parametres)

        # Pour chaque ligne - une page A4 PAYSAGE avec les deux graphiques côte à côte
        for i, (idx, row) in enumerate(df.iterrows()):
            quadrilateres = graphs["Graph"][idx]
            fig_complete = creer_page_complete(row, quadrilateres, page_num=i + 3, total_pages=total_pages)
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


def Ajout_Titre(input_pdf, watermark_url, transparency, scale, pos_y, pos_x):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    # Télécharger l'image depuis GitHub (raw URL)
    response = requests.get(watermark_url)
    img_data = io.BytesIO(response.content)
    img = ImageReader(img_data)

    # Dimensions originales de l'image
    orig_width, orig_height = img.getSize()

    for i, page in enumerate(reader.pages):
        # Ne pas appliquer sur la première page
        if i == 0:
            writer.add_page(page)
            continue

        largeur = float(page.mediabox.width)
        hauteur = float(page.mediabox.height)

        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=(largeur, hauteur))

        x = pos_x * largeur
        y = pos_y * hauteur

        c.setFillAlpha(transparency)

        # Conserver proportions : multiplier largeur et hauteur originales par le facteur scale
        img_width = orig_width * scale
        img_height = orig_height * scale

        # Centrer l'image autour de (x, y)
        c.drawImage(img, x - img_width/2, y - img_height/2,
                    width=img_width, height=img_height, mask='auto')

        c.save()

        packet.seek(0)
        watermark = PdfReader(packet)
        page.merge_page(watermark.pages[0])
        writer.add_page(page)

    output = io.BytesIO()
    writer.write(output)
    output.seek(0)
    return output

def colorer_valeurs(val):
    if val > 0:
        color = "green"
    elif val < 0:
        color = "red"
    else:
        color = "orange"
    return f"color: {color}; font-weight: bold;"
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

url = 'https://github.com/lilianmtech/Analyse_Parallelogramme_Vitrage/blob/main/'

# Configuration de la page
st.set_page_config(page_title="Compatibilité des déformations du support avec l’intégrité des vitrages", layout="wide")

    #---------------------------Importation des données---------------------------

st.sidebar.image(url+"logo-couleur.png?raw=true",width=200)

  # Section d'import de fichier Excel (commune aux deux onglets)
st.sidebar.header("📁 Import de données")
uploaded_file = st.sidebar.file_uploader("Importer un fichier de données (CSV ou Excel)", type=["csv", "xlsx"])


# Titre de l'application
st.markdown(
    """
    <h2 style='text-align: center; 
            color: #008A92; 
            font-family: Verdana; 
            font-size:30px;
            background-color: white;
            padding: 20px; 
            border-radius: 1px;'>
        Compatibilité des déformations du support avec l’intégrité des vitrages
    </h2>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    /* Style du texte des onglets */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.7rem;
        font-weight: 700;
        color: #495556;
    }

    /* Centrage des onglets */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 50px;
    }
            
    /* Couleur du trait de sélection */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #17A2A8;
        height: 4px; /* épaisseur du trait */
        border-radius: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  Analyse de la mise en parallélogramme  ", "  Analyse du gauchissement  "])

with tab1:
    st.markdown("### 📏 Défitinion des jeux et tolérances")
    Typo = st.selectbox("Typologie :", ["Façade", "Verrière"])
    choix = st.selectbox("Calage latéral :", ["Sans", "Avec"])

    # --- Ligne 1 ---
    ligne1_col1, ligne1_col2 = st.columns(2)
   
    with ligne1_col1:
        if choix == "Sans":
            st.image(url+"Borne_M.png"+'?raw=true', caption="Bornes sur montants", width=450)
        elif choix == "Avec":
            st.image(url+"Borne_M_C.png"+'?raw=true', caption="Bornes sur montants", width=450)


    with ligne1_col2:
        if choix == "Sans":
            st.image(url+"Borne_T.png"+'?raw=true', caption="Bornes sur traverses", width=400)
        elif choix == "Avec":
            st.image(url+"Borne_T_C.png"+'?raw=true', caption="Bornes sur traverses", width=400)   

    # --- Ligne 2 ---
    ligne2_col1, ligne2_col2 = st.columns(2)

    with ligne2_col1:
        if choix == "Sans":
            Tm1 = st.number_input("Valeur Tolérance Tm1 :", value=1.95, step=0.1, format="%.2f")
            Tm2 = st.number_input("Valeur Tolérance Tm2 + menuiserie :", value=9.95, step=0.1, format="%.2f")
        else:
            Tm =  st.number_input("Valeur Tolérance Tm :", value=1.95, step=0.1, format="%.2f")
            Ca =  st.number_input("Epaisseur cale C + menuiserie :", value=10.0, step=0.1, format="%.2f")
            Jc =  st.number_input("Jeu entre vitrage et cale Jc :", value=1.5, step=0.1, format="%.2f")
        
    with ligne2_col2:
        if choix == "Sans":
            Tt1 = st.number_input("Valeur Tolérance Tt1 :", value=3.90, step=0.1, format="%.2f")
            Tt2 = st.number_input("Valeur Tolérance Tt2 + menuiserie :", value=11.90, step=0.1, format="%.2f")
        else:
            Tt1 =  st.number_input("Valeur Tolérance Tt1 :", value=3.90, step=0.1, format="%.2f")
            Tt2 =  st.number_input("Valeur Tolérance Tt2 + menuiserie :", value=11.90, step=0.1, format="%.2f")
            Jv =  st.number_input("Jeu entre borne basse et vitrage Jv :", value=0.9, step=0.1, format="%.2f")

    #---------------------------Chargement des vitrages---------------------------
    if choix == "Sans":
        Raico = [Tm1, Tm2, Tt1, Tt2]

    else:
        Raico = [Tm, Ca, Jc, Tt1, Tt2, Jv]


    if uploaded_file:
        # Le contenu du fichier (pas seulement son nom) détermine si les
        # données doivent être recalculées : un fichier réimporté sous le
        # même nom mais avec un contenu modifié doit déclencher une mise à jour.
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

        if "uploaded_hash" not in st.session_state or st.session_state["uploaded_hash"] != file_hash:
            st.session_state.clear()
            st.session_state["uploaded_hash"] = file_hash

        # Vérifier si un changement de paramètre nécessite une mise à jour
        params_actuels = {
            "Typo": Typo,
            "choix": choix,
            "Raico": Raico,
            "file_hash": file_hash,
        }

        if (
            "params_prec" not in st.session_state
            or st.session_state["params_prec"] != params_actuels
        ):
            datas, graphs = Dechaussement_vitrage(uploaded_file, choix, Raico, Typo)
            st.session_state["datas"] = datas
            st.session_state["graphs"] = graphs
            st.session_state["params_prec"] = params_actuels
        else:
            datas = st.session_state["datas"]
            graphs = st.session_state["graphs"]

        # -------------------- Affichage du tableau --------------------
        st.divider()
        st.markdown("### 📊 Tableau des Résultats")
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
        st.divider()
        st.markdown("### 🔲 Sélection et Visualisation")
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
                    if col != 'ID vitrage' and col != 'Gamme' and col != 'Pf (mm)' and col != 'Demi-périmètre (m)':
                        st.metric(col, ligne_data[col])
                        st.markdown(
                            """
                            <style>
                            /* Cible la valeur affichée dans st.metric */
                            div[data-testid="stMetricValue"] {
                                font-size: 20px;   /* taille plus petite */
                            }

                            /* Cible le label (titre) de la métrique */
                            div[data-testid="stMetricLabel"] {
                                font-size: 18px;   /* taille plus petite pour le label */
                            }

                            /* Cible la variation (delta) */
                            div[data-testid="stMetricDelta"] {
                                font-size: 12px;   /* taille plus petite pour le delta */
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

        with col2:
            idx = graphs["ID vitrage"].index(ligne_selectionnee)
            quadrilateres = graphs["Graph"][idx]
            fig = visualiser_quadrilateres(quadrilateres)
            st.pyplot(fig)

        # Informations supplémentaires sur le déplacement de la traverse haute / du montant
        cadre = quadrilateres[0]
        cadre_ref = quadrilateres[4]
        st.info(f"📐 {texte_deplacement_relatif(cadre, cadre_ref)}")

        
        st.divider()
        col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
        with col_pdf2:
            st.subheader("📄 Rapport PDF Complet")
            st.write("Téléchargez un rapport PDF contenant les détails et visualisations de toutes les lignes")
            titre_personnalise = st.text_input(
                "Titre du rapport (optionnel)",
                value="",
                placeholder="Ex : Résidence Les Ormes - Lot 3",
                help="Affiché sur la page de garde, sous le titre principal."
            )
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #17A2A8; /* bleu */
                    color: white;              /* texte en blanc */
                    border-radius: 8px;
                    padding: 10px 20px;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Bouton pour générer le PDF
            if st.button("🔄 Générer le Rapport PDF", use_container_width=True, type="primary"):
                with st.spinner("Génération du rapport PDF en cours..."):
                    try:
                        pdf = generer_rapport_pdf(df_affichage, graphs, Typo, choix, Raico, titre_personnalise=titre_personnalise)
                        pdf_buffer=Ajout_Titre(pdf, 'https://github.com/lilianmtech/Analyse_Parallelogramme_Vitrage/blob/main/logo-couleur.png?raw=true', 0.3, 0.3, 0.08, 0.9)
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
                    file_name="rapport_parallelo_vitrage.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_button"
                )
                st.info(f"📋 Le rapport contient {len(df_affichage) + 1} pages")

    else:
        st.sidebar.info("📥 Importez un fichier Excel pour commencer l’analyse.")
            # Footer
    st.caption("Application développée avec Streamlit et Shapely")

with tab2:
    if uploaded_file:
        # Le gauchissement ne dépend que du contenu du fichier : on ne recalcule
        # que si celui-ci a changé (et on ne touche pas au cache de l'autre onglet)
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if (
            "gauchissement_hash" not in st.session_state
            or st.session_state["gauchissement_hash"] != file_hash
        ):
            st.session_state["gauchissement_datas"] = Gauchissement_vitrage(uploaded_file)
            st.session_state["gauchissement_hash"] = file_hash

        datas = st.session_state["gauchissement_datas"]

        # -------------------- Affichage du tableau --------------------
        st.markdown("### 📊 Tableau des Résultats")
        df_affichage = pd.DataFrame(datas).round(1)

        # Fonction pour colorer la colonne Gauchissement
        def color_gauchissement(colonne):
            styles = []
            for i in range(len(colonne)):
                if df_affichage.loc[i, 'Gauchissement (mm)'] < df_affichage.loc[i, 'Critère cahier 3574v2 (mm)*'] and df_affichage.loc[i, 'Gauchissement (mm)'] < df_affichage.loc[i, 'Critère DTU39-P4 (mm)**']:
                    styles.append("color: green;text-align: center; font-size: 18px;")
                else:
                    styles.append("color: red;text-align: center; font-size: 18px")
            return styles

        # Fonction pour griser la colonne Critère
        def gray_critere(colonne):
            return ["color: gray;" for _ in colonne]

        # Application des styles
        styled_df = (
            df_affichage.style
            .apply(color_gauchissement, subset=['Gauchissement (mm)'])
            .apply(gray_critere, subset=['L (mm)','Diag (mm)','Critère cahier 3574v2 (mm)*','Critère DTU39-P4 (mm)**'])
            .format({
                'Gauchissement (mm)': "{:.1f}",
                'L (mm)': "{:.1f}",
                'Diag (mm)': "{:.1f}",
                'Critère cahier 3574v2 (mm)*': "{:.1f}",
                'Critère DTU39-P4 (mm)**': "{:.1f}"
            })
            .set_properties(**{"text-align": "center", "font-size": "18px"})
        )
        st.dataframe(styled_df, width='stretch', hide_index=True)
        st.info("""\* Critère admissible suivant le tableau 11 du cahier du CSTB 3574v2 : LPetit Côté/75""")
        st.info("""\** Critère admissible suivant le §9.2 du DTU39-P4 : Diag/150""")
        st.info("""❕ Critère valable pour vitrages isolants""")
