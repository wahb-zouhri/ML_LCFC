# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:09:42 2022

@author: Wahb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from heatmap import corrplot
from sklearn.decomposition import PCA
from os import remove
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
    
# In[0] - défintion de la classe ACP
                               
class ACP(object):

    def __init__(self, data):
        self.data = data
        self.pca = PCA()
        
    def normaliser(self):
        """
        En théorie des probabilités et en statistique, une variable centrée réduite est la transformée d'une variable aléatoire par une application, de telle sorte que sa moyenne soit nulle <x_moy=0> et son écart type égal à un <𝜎=1>.

        Returns
        -------
        numpy array:
            Données centrées réduites.

        """
        sc = StandardScaler()
        return sc.fit_transform(self.data)
    
    def transformer(self):
        """
        Renvoie la matrice de données projetées dans le nouvel espace défini par les axes principaux.

        Returns
        -------
        numpy array:
            Données transformées dans la nouvelle base.

        """
        return self.pca.fit_transform(self.normaliser())  
    
    def valeurs_propres(self):
        """
        Renvoie un vecteur contenant l'ensemble des valeurs propres.

        Returns
        -------
        numpy array:
            Valeurs propres.

        """
        self.transformer()
        Len = self.data.shape[0]
        return ((Len-1)/Len)*self.pca.explained_variance_ 
    
    def vecteurs_propres(self):
        """
        Renvoie une matrice où chaque ligne représente un vecteur propre.

        Returns
        -------
        numpy array:
            Vecteurs propres.

        """
        self.transformer()
        return self.pca.components_
    
    def inertie(self):
        """
        Renvoie l'inertie (taux d'information) liée à chaque axe principal.

        Returns
        -------
        numpy array:
            Inertie.

        """
        self.transformer()
        return self.pca.explained_variance_ratio_
    
    def inertie_cumulative(self):
        """
        Renvoie un vecteur contenant l'inertie cumulative. Par exepmle : le 2ème élément du vecteur représente la somme des inerties liées aux 1er et 2ème axes principaux.

        Returns
        -------
        numpy array:
            Inertie cumulative

        """
        Inertie_sum = self.inertie()   
        
        for i in range(1,self.data.shape[1]):
            Inertie_sum[i] = Inertie_sum[i] + Inertie_sum[i-1]
        return Inertie_sum
    
    def corr_axe_param(self):
        """
        Renvoie la matrice de corrélartion entre les paramètres/variables et les axes principaux.

        Returns
        -------
        pandas DataFrame:
            Corrélations entre les paramètres et les axes principaux.

        """
        dim = self.data.shape[1]
        mat_cor = np.zeros((dim,dim))    
        
        for k in range(dim):
            mat_cor[:,k] = self.vecteurs_propres()[k,:]*self.valeurs_propres()[k]**0.5
        return pd.DataFrame(data = mat_cor, index = self.data.columns, columns = ["Axe"+str(i+1) for i in range(dim)])
        
    def corr_param_param(self):
        """
        Renvoie la matrice de corrélartion entre les différents paramètres.

        Returns
        -------
        pandas DataFrame:
            matrice de corrélation.

        """
        return self.data.corr()
    
    def plot_valeurs_propres(self):
        """
        Générer une visualisation des différentes valeurs propres et de la variation cumulative.

        """
        
        dim = self.data.shape[1]
        fig, ax1 = plt.subplots(figsize=(10,6))
    
        X_labels =  ['λ'+str(i+1) for i in range(dim)]
        ax1.bar(X_labels, height = self.valeurs_propres(), color='green')
        ax1.grid(axis='y')
        plt.xlabel("Composantes pricipales")
        plt.ylabel("Valeurs propres")
        
        ax2=ax1.twinx()       # pour superposer deux grapghes
        ax2.plot(self.inertie_cumulative(),color='red')
        for i in range(0,dim):
            ax2.annotate(("{:.2%}".format(self.inertie_cumulative()[i])), (i,self.inertie_cumulative()[i]), size=12)  # 1er argument = l'annotation; 2eme = le point (xi,yi)
        ax2.scatter(np.arange(0,dim), self.inertie_cumulative(), s=20, color='black')
        plt.ylabel("Variation cumulative")
        
        plt.show()
        return fig
    
    def plot_matrice_corr(self):
        """
        Générer une visualisation de la matrice de corrélation d'un jeu de données.

        """
        size = int(self.data.shape[1]*0.75) 
        fig02 = plt.figure(figsize=(size, size))
        corrplot(self.corr_param_param(), size_scale=1100, marker="o")
        return fig02

    def plot_projection(self, axe_i, axe_j, param_gradient):
        """
        la fonction permet de : 
            +) visualiser les données dans un plan 2D défini par les deux axes principaux 'axe_i', 'axe_j' et par le gradient de couleur correspondant au paramètre 'param_gradient'.\n
            +) visualiser les paramètres d'entrée sur ce plan 2D afin de mettre en exergue les différentes corrélations.        

        Parameters
        ----------
        axe_i : int
            1er axe principal qui définit le plan 2D.
        axe_j : int
            2ème axe principal qui définit le plan 2D.
        param_gradient : str
            nom du paramètre.

        """
        Tr_data = self.transformer()
        dim = self.data.shape[1]
        Corr_ax_par = self.corr_axe_param()

        axe_i = int(axe_i-1); axe_j = int(axe_j-1)
        assert type(param_gradient) == str, "'param_gradient' doit être une chaîne de caractères."
        
        fig= plt.figure(figsize=(8,18))
        
        ax1 = fig.add_subplot(212)
        ax1.spines['top'].set_color('none')       # 'top' arête sans couleur
        ax1.spines['right'].set_color('none')
        ax1.spines['bottom'].set_position('zero') # positionner 'bottom' arête au niveau du zéro 
        ax1.spines['left'].set_position('zero')
        
        plt.scatter(Tr_data[:,axe_i],Tr_data[:,axe_j],s=7.5, c=data[param_gradient], cmap='Reds')
        plt.title("Plan {}x{}: Inertie {:.2%}".format(axe_i+1, axe_j+1, self.inertie()[axe_i]+self.inertie()[axe_j]), size=14)
        plt.grid(linestyle=':')
        
        ax2 = fig.add_subplot(211)
        ax2.spines['top'].set_color('none')       
        ax2.spines['right'].set_color('none')
        ax2.spines['bottom'].set_position('zero') 
        ax2.spines['left'].set_position('zero')
        C = plt.Circle((0,0),1,color='r',fill=False)
        ax2.add_artist(C)
        
        # Définir les flèches
        O=np.zeros((dim,1))
        plt.quiver(O, O, Corr_ax_par.iloc[:,axe_i], Corr_ax_par.iloc[:,axe_j],angles='xy', scale_units='xy', scale=1, color='blue', width = 0.003, headwidth=1)
        ax2.scatter(Corr_ax_par.iloc[:,axe_i], Corr_ax_par.iloc[:,axe_j],s=20,color='red')
        for i in range(0,dim):
            ax2.annotate(self.data.columns[i], (Corr_ax_par.iloc[i,axe_i], Corr_ax_par.iloc[i,axe_j]), size=12, color = 'red')
        
        plt.xlim((-1.05,1.05)); plt.ylim((-1.05,1.05))
        plt.grid(linestyle=':')    
        plt.title("Plan {}x{}: Inertie {:.2%}".format(axe_i+1, axe_j+1, self.inertie()[axe_i]+self.inertie()[axe_j]), size=14)
        
        plt.show()
        return fig
    
    def enregistrer_excel(self, nom_fig1 , nom_fig2, nom_fig3):
        """
        
        Parameters
        ----------
        nom_fig1 : str
            nom du fichier jpg enregistré contenant la visualisation des valeurs propres.
        nom_fig2 : str
            nom du fichier jpg enregistré contenant le cercle de corrélation et la projection des données.
        nom_fig3 : str
            nom du fichier jpg enregistré contenant la matrice de corrélation.
        nom_fichier : str, optional
            nom du fichier excel à générer. La valeur par défaut est 'Résultats_ACP'.

        Returns
        -------
        Fichier excel synthétisant les différents résultats de l'ACP.

        """
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')

        self.data.to_excel(writer, sheet_name='données initiales')
        
        N_data = pd.DataFrame(data=self.normaliser(), columns=self.data.columns)
        N_data.to_excel(writer, sheet_name='données normalisées')
        
        Tr_data = pd.DataFrame(data=self.transformer(), columns=["Axe"+str(i+1) for i in range(self.data.shape[1])])
        Tr_data.to_excel(writer, sheet_name='données transformées')
        
        acp_results = pd.DataFrame(data=np.vstack((self.valeurs_propres(),self.inertie(),self.inertie_cumulative())), 
             index=["Valeur propre", "Variation", "Variation cumulative"], columns=["Axe"+str(i+1) for i in range(data.shape[1])]).T
        acp_results.to_excel(writer, sheet_name='Valeurs propres et Inertie')
        worksheet = writer.sheets['Valeurs propres et Inertie']
        worksheet.insert_image('G1', nom_fig1+'.jpg', {'x_scale': 0.8, 'y_scale': 0.8})
        
        Corr_ax_par.to_excel(writer, sheet_name='cercle de corrélations', startcol=8)
        worksheet = writer.sheets['cercle de corrélations']
        worksheet.insert_image('A1', nom_fig2+'.jpg', {'x_scale': 0.7, 'y_scale': 0.7})
        
        Corr_par_par.to_excel(writer, sheet_name='corrélations entre paramètres', startcol=11)
        worksheet = writer.sheets['corrélations entre paramètres']
        worksheet.insert_image('A1', nom_fig3+'.jpg', {'x_scale': 0.8, 'y_scale': 0.8})
        
        writer.save()
        xlsx_data = output.getvalue()
                
        return xlsx_data
  
        
# In[0] - streamlit layout

@st.cache
def get_data(path, sep=';'):
    if path is not None:
        try:           
            data = pd.read_csv(path, header=0, sep =str(sep))
        except:
            data = pd.read_excel(path, header=0)               
    else:
        data =  pd.DataFrame(data = 10*np.random.rand(1000,10), columns = ["P"+str(i) for i in range(1,11)]) # random data so the app can run when no data are provided
    
    return data

intro = st.container()
dataset = st.container()
inertie = st.container()
correlation = st.container()
projection = st.container()
save = st.container()


with intro:
    st.title('Analyse en Composantes Principales (ACP)')
    
with dataset:
    st.header("Importation du jeu de données")
    col1,_ = st.columns(2)
    sep = col1.text_input("Veuillez spécifier le séparateur utilisé si vous souhaitez importer un fichier CSV. ", value = ';')
    data = st.file_uploader("Importer ici votre fichier EXCEL ou CSV.", type =['csv','xlsx'])
    data = get_data(data)
    data = pd.DataFrame(data)
    st.write(data.head(-1))

with inertie:
    st.header("Résultats de l'ACP")
    st.subheader("Inertie et valeurs propores")

    acp = ACP(data)
    val_pr = acp.valeurs_propres()
    var = acp.inertie()
    var_cum = acp.inertie_cumulative()
    acp_results = pd.DataFrame(data=np.vstack((val_pr,var,var_cum)), 
             index=["Valeur propre", "Variation", "Variation cumulative"], columns=["Axe"+str(i+1) for i in range(data.shape[1])])
    fig01 = acp.plot_valeurs_propres()
    
    st.write(acp_results)
    st.pyplot(fig01)
    
with correlation:
    Corr_ax_par = acp.corr_axe_param()
    Corr_par_par = acp.corr_param_param()
    fig03 = acp.plot_matrice_corr()
    
    st.subheader("Matrice de corrélation")
    st.write(Corr_par_par)
    st.pyplot(fig03)
    
with projection:
    st.subheader("Projection des données et des variables")
    inputs, outputs = st.columns([1.1,4])
    axe_1 = inputs.selectbox('Premier axe principal', range(1,data.shape[1]+1))
    axe_2 = inputs.selectbox('Deuxième axe principal', range(1,data.shape[1]+1), index = 1)
    grad_col = inputs.selectbox('Paramètre définissant le gradient de couleur', data.columns)
    
    fig02 = acp.plot_projection(int(axe_1), int(axe_2), grad_col)
    outputs.pyplot(fig02)

with save:
    st.subheader("Exporter les résultats de l'ACP sous format Excel")

    fig01.savefig('fig01.jpg',format='jpg', dpi=300, bbox_inches = 'tight', pad_inches=0.1)        # nom de fichier modifiable 
    fig03.savefig('fig03.jpg',format='jpg', dpi=300, bbox_inches = 'tight', pad_inches=0.1)        # nom de fichier modifiable 
    fig02.savefig('fig02.jpg',format='jpg', dpi=300, bbox_inches = 'tight', pad_inches=0.1)        # nom de fichier modifiable 
    out_data = acp.enregistrer_excel('fig01', 'fig02', 'fig03') 
    remove('fig01.jpg'); remove('fig02.jpg'); remove('fig03.jpg')
    
    st.download_button(label='📥 Enregistrer sous', data = out_data, file_name = 'Résultats ACP.xlsx')

    
 