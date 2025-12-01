import sympy as sy
from math import *
from numpy import *
from shapely import *
from scipy.optimize import minimize_scalar

class Vitrage:
    def __init__(self,cadre_0,cadre_def,Gamme,raico,pf,calage_lateral='Sans'):
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
        self.p = pf
        self.raico = raico

        self.cale_bas = 100
        self.cale_laterale = 100

    def DiffHorsPlan(self):
        "Vecteur normal"
        a = sy.Symbol('a')
        b = 1
        c = sy.Symbol('c')
        
        "Coeffiecient directeur de AB"
        x1 = self.B[0]-self.A[0]
        y1 = self.B[1]-self.A[1]
        z1 = self.B[2]-self.A[2]
        
        "Coeffiecient directeur de AD"
        x2 = self.D[0]-self.A[0]
        y2 = self.D[1]-self.A[1]
        z2 = self.D[2]-self.A[2]
        
        "Détermination des coef du vecteur normal"
        eq1 = sy.Eq(a*x1 + b*y1 + c*z1 , 0)
        eq2 = sy.Eq(a*x2 + b*y2 + c*z2 , 0)
        sol1 = sy.solve((eq1,eq2),(a,c))
        
        if sol1 == []:
            distance = 'None'
        else:
            a=float(sol1[a])
            c=float(sol1[c])
            
            "Détermination des coef de l'équation de plan"
            d = -a*self.A[0]-b*self.A[1]-c*self.A[2]
            
            "Distance entre point et plan"
            distance = abs(a*self.C[0]+b*self.C[1]+c*self.C[2]+d)/sqrt(a**2+b**2+c**2)
            
        #-----------Distance entre le sommet et le centre de la diagonale---------
        X = sy.Symbol('X')
        
        O = []
        
        "Coefficient droite AC"
        a1 = (self.C[1]-self.A[1])/(self.C[0]-self.A[0])
        b1 = self.C[1] - (a1*self.C[0])
        
        "Coefficient droite BD"
        a2 = (self.D[1]-self.B[1])/(self.D[0]-self.B[0])
        b2 = self.D[1] - (a2*self.D[0])
        
        eq3 = sy.Eq(a1*X + b1 - a2*X - b2 , 0)
        sol3 = sy.solve(eq3,X)
        O.append(float(sol3[0]))
        O.append(a1*O[0]+b1)
        
        LO = sqrt((O[0]-self.C[0])**2+(O[1]-self.C[1])**2)
        
        if abs(distance)<abs(LO/75):
            Gauchissement = 'OK'
        else:
            Gauchissement = 'NOK'

        return distance
    
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
        
        if self.p > 7.0 :
            pf = 12.0
        elif  5.0 < self.p <= 7.0 :
            pf = 9.0
        elif self.p <= 5.0 :
            pf = 6.0
        
        if self.calage_lateral == 'Sans':
            Tm1 = self.raico[0]
            Tm2 = self.raico[1]
            Tt1 = self.raico[2]
            Tt2 = self.raico[3]

            pos_vit_lat = (self.Gamme/2-(Tm1+pf+Tm2))/2+Tm2
            pos_vit_haut = (self.Gamme/2-(Tt1+pf+Tt2))/2+Tt2
            jeu_borne_lat = (self.Gamme/2-(Tm1+pf+Tm2))/2
            jeu_borne_haut = (self.Gamme/2-(Tt1+pf+Tt2))/2

            donnees = (pos_vit_lat, pos_vit_haut, jeu_borne_lat, jeu_borne_haut, 0)

        elif self.calage_lateral == 'Avec':
            Tm = self.raico[0]
            Ca = self.raico[1]
            Jc = self.raico[2]
            Tt1 = self.raico[3]
            Tt2 = self.raico[4]
            Jv = self.raico[5]
        
            pos_vit_lat = Ca+Jc
            pos_vit_haut = self.Gamme/2-(Tt2+pf+Tt1+Jv)+Tt2
            jeu_borne_lat = self.Gamme/2 - (Ca+Jc+pf+Tm)
            jeu_borne_haut = self.Gamme/2-(Tt1+pf+Tt2+Jv)

            donnees = (pos_vit_lat, pos_vit_haut, jeu_borne_lat, jeu_borne_haut, Ca)

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

        'Translation du vitrage'
        Vitrage_T = affinity.translate(Vitrage,xoff=0.0, yoff=-support_bas['T_'+cote_support_bas], zoff=0.0)
        'Rotation du vitrage'
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

        if self.calage_lateral == 'Avec':
            Borne_ext = self.inner_quad_with_offsets( Cadre, offset_top=Raico['Pos_vit_haut']-Raico['Tolerance traverse'], offset_bottom=13.0,offset_lr=Raico['Calage lateral'])
            Borne_int = self.inner_quad_with_offsets( Cadre, offset_top=Raico['Pos_vit_haut']+Jv, offset_bottom=13.0,offset_lr=Raico['Pos_vit_lat']+Raico['Tolerance epine'])
        else :
            Borne_ext = self.inner_quad_with_offsets( Cadre, offset_top=Raico['Pos_vit_haut']-Raico['Tolerance traverse'], offset_bottom=13.0,offset_lr=Raico['Pos_vit_lat']-Raico['Tolerance epine'])
            Borne_int = self.inner_quad_with_offsets( Cadre, offset_top=Raico['Pos_vit_haut']+Raico['Tolerance traverse'], offset_bottom=13.0,offset_lr=Raico['Pos_vit_lat']+Raico['Tolerance epine'])


        A_BE, B_BE, C_BE, D_BE, ini = list(Borne_ext.exterior.coords)
        A_BI, B_BI, C_BI, D_BI, ini = list(Borne_int.exterior.coords)

        Ctrl_gauche_BE = A_V[0] - A_BE[0]
        Ctrl_gauche_BI = A_BI[0] - A_V[0]
        Ctrl_droite_BE = B_BE[0] - B_V [0]
        Ctrl_droite_BI =  B_V[0] - B_BI[0]
        Ctrl_haut_BE = min(A_BE[1] - A_V [1],B_BE[1] - B_V [1])
        Ctrl_haut_BI = min( A_V [1] - A_BI[1],B_V [1] - B_BI[1])
        
        if self.calage_lateral == 'Avec' :
            if contact_cale == 'droite':
                Ctrl_droite_BE = 0

            elif contact_cale == 'gauche':
                Ctrl_gauche_BE = 0

        Vitrage_T_R = Polygon(Vitrage_T_R)
        
        data = [round(min(Ctrl_gauche_BE,Ctrl_gauche_BI),1),round(min(Ctrl_droite_BE,Ctrl_droite_BI),1),round(min(Ctrl_haut_BE,Ctrl_haut_BI),1)]
        graph = [Cadre, Borne_ext, Vitrage_T_R, Borne_int]
        
        return data, graph