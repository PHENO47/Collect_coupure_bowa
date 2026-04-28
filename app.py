import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(
    page_title="Collecte Coupure BOWA - Analyse des coupures électriques",
    page_icon="⚡",
    layout="wide"
)

# ==================== BARRE LATÉRALE (STYLE IMAGE) ====================
with st.sidebar:
    # Logo / Titre
    st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <h2 style='color: #FF6B35; margin-bottom: 0;'>⚡ BOWA</h2>
            <p style='color: #666; font-size: 0.8rem;'>Collecte Coupure</p>
            <hr>
        </div>
    """, unsafe_allow_html=True)
    
    # Menu de navigation
    st.markdown("### 📌 Menu")
    menu = st.radio(
        "",
        [ "📝 Nouveau signalement", "📋 Données brutes", "📊 Tableau de bord", "📈 Analyses", "⚙️ À propos"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Barre de recherche (comme sur l'image)
    #st.markdown("### 🔍 Rechercher")
    #search = st.text_input("", placeholder="Rechercher un signalement...", label_visibility="collapsed")
    
    #st.markdown("---")
    
    # Profil utilisateur (comme sur l'image)
    st.markdown("""
        <div style='background: #f0f2f6; padding: 12px; border-radius: 10px;'>
            <p style='margin: 0; font-weight: bold;'>👤 Gov BOWA</p>
            <p style='margin: 0; color: #666; font-size: 0.75rem;'>Supervision</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats rapides dans la sidebar
    fichier_data = "data/coupures.csv"
    if os.path.exists(fichier_data):
        df_temp = pd.read_csv(fichier_data)
        st.markdown("---")
        st.markdown("### 📊 Stats rapides")
        st.metric("Total signalements", len(df_temp), delta=None)

# ==================== CONTENU PRINCIPAL ====================

# En-tête
st.markdown("""
    <div style='margin-bottom: 20px;'>
        <h1 style='color: #FF6B35; margin-bottom: 0;'>Collecte Coupure BOWA</h1>
        <p style='color: #666;'>Système de collecte et analyse des coupures d'électricité</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== PAGE : TABLEAU DE BORD ====================
if menu == "📊 Tableau de bord":
    
    st.markdown("## 📊 Tableau de bord")
    
    fichier_data = "data/coupures.csv"
    
    if os.path.exists(fichier_data):
        df = pd.read_csv(fichier_data)
        
        # Nettoyage
        df["duree_heures"] = pd.to_numeric(df["duree_heures"], errors="coerce")
        df["impact_numerique"] = pd.to_numeric(df["impact_numerique"], errors="coerce")
        df_clean = df.dropna(subset=["duree_heures"])
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total incidents", len(df_clean))
        with col2:
            st.metric("⏱️ Durée moyenne", f"{df_clean['duree_heures'].mean():.1f} h")
        with col3:
            st.metric("⚠️ Cause principale", df_clean['cause'].mode().iloc[0] if len(df_clean) > 0 else "N/A")
        with col4:
            st.metric("📍 Zone la plus touchée", df_clean['zone'].mode().iloc[0] if len(df_clean) > 0 else "N/A")
        
        st.markdown("---")
        
        # Graphiques du tableau de bord
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Évolution des signalements")
            if 'timestamp' in df_clean.columns:
                df_clean['date'] = pd.to_datetime(df_clean['timestamp']).dt.date
            signalements_par_jour = df_clean.groupby(df_clean['date']).size() if 'date' in df_clean.columns else pd.Series()
            if len(signalements_par_jour) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                signalements_par_jour.plot(kind='line', marker='o', ax=ax, color='#FF6B35')
                ax.set_xlabel("Date")
                ax.set_ylabel("Nombre de signalements")
                ax.set_title("Signalements par jour")
                st.pyplot(fig)
            else:
                st.info("Pas assez de données")
        
        with col2:
            st.markdown("### 🎯 Répartition par zone")
            zone_counts = df_clean['zone'].value_counts().head(5)
            if len(zone_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                zone_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_xlabel("Zone")
                ax.set_ylabel("Nombre d'incidents")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚡ Causes des coupures")
            cause_counts = df_clean['cause'].value_counts()
            if len(cause_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                cause_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#FF6B35', '#FFB347', '#FFD700', '#87CEEB', '#98FB98'])
                ax.set_ylabel("")
                ax.set_title("Répartition des causes")
                st.pyplot(fig)
        
        with col2:
            st.markdown("### 📊 Types de zones touchées")
            type_counts = df_clean['type_zone'].value_counts()
            if len(type_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                type_counts.plot(kind='bar', ax=ax, color='#FF6B35')
                ax.set_xlabel("Type de zone")
                ax.set_ylabel("Nombre d'incidents")
                st.pyplot(fig)
        
    else:
        st.info("ℹ️ Aucune donnée disponible. Utilisez le menu '📝 Nouveau signalement' pour commencer.")

# ==================== PAGE : DONNÉES BRUTES ====================
elif menu == "📋 Données brutes":
    
    st.markdown("## 📋 Données brutes")
    
    fichier_data = "data/coupures.csv"
    
    if os.path.exists(fichier_data):
        df = pd.read_csv(fichier_data)
        
        # Filtre de recherche
        #if search:
            #mask = df['ville'].str.contains(search, case=False, na=False) | df['commentaire'].str.contains(search, case=False, na=False)
            #df_filtered = df[mask]
            #st.info(f"🔍 Résultats pour : '{search}' - {len(df_filtered)} signalement(s) trouvé(s)")
            #st.dataframe(df_filtered, use_container_width=True)
        #else:
            #st.dataframe(df, use_container_width=True)
        
        #st.markdown("---")
        
        # Export
        st.download_button(
            label="📥 Télécharger toutes les données (CSV)",
            data=df.to_csv(index=False),
            file_name=f"signalements_coupures_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Aucune donnée disponible")

# ==================== PAGE : NOUVEAU SIGNALEMENT ====================
elif menu == "📝 Nouveau signalement":
    
    st.markdown("## 📝 Nouveau signalement de coupure")
    
    with st.form("signalement_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ville = st.text_input("Ville / Quartier *", placeholder="Ex: Douala, Yaoundé, Garoua...")
            zone = st.selectbox("Zone géographique", [
                "Centre", "Littoral", "Ouest", "Nord", "Extrême-Nord", 
                "Sud", "Sud-Ouest", "Est", "Nord-Ouest", "Adamaoua"
            ])
            type_zone = st.selectbox("Type de zone", ["Urbaine", "Péri-urbaine", "Rurale"])
            duree_heures = st.slider("Durée de la coupure (heures)", 0.5, 48.0, 4.0, 0.5)
            
        with col2:
            cause_probable = st.selectbox("Cause probable", [
                "Pluie/orage", "Vent violent", "Surcharge réseau", 
                "Travaux programmés", "Accident (véhicule/câble)", 
                "Vol/câble volé", "Arbre tombé", "Inconnue"
            ])
            frequence = st.selectbox("Fréquence dans cette zone", [
                "Première fois", "Rare (1-2x/mois)", "Occasionnelle (1x/semaine)", 
                "Fréquente (2-3x/semaine)", "Quotidienne"
            ])
            impact = st.select_slider("Personnes touchées estimées", 
                options=["Moins de 50", "50-200", "200-500", "500-2000", "Plus de 2000"])
            commentaire = st.text_area("Observations", placeholder="Informations supplémentaires...")
        
        submitted = st.form_submit_button("✅ Enregistrer le signalement", use_container_width=True)
        
        impact_map = {
            "Moins de 50": 25, "50-200": 125, "200-500": 350,
            "500-2000": 1250, "Plus de 2000": 3000
        }
        frequence_map = {
            "Première fois": 0.1, "Rare (1-2x/mois)": 1.5,
            "Occasionnelle (1x/semaine)": 4, "Fréquente (2-3x/semaine)": 10,
            "Quotidienne": 30
        }
        
        if submitted:
            if not ville:
                st.error("Veuillez renseigner la ville/quartier")
            else:
                fichier_data = "data/coupures.csv"
                os.makedirs("data", exist_ok=True)
                
                nouvelle_ligne = pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ville": ville,
                    "zone": zone,
                    "type_zone": type_zone,
                    "duree_heures": duree_heures,
                    "cause": cause_probable,
                    "frequence": frequence,
                    "impact": impact,
                    "commentaire": commentaire,
                    "impact_numerique": impact_map[impact],
                    "frequence_numerique": frequence_map[frequence]
                }])
                
                if os.path.exists(fichier_data):
                    df_existant = pd.read_csv(fichier_data)
                    df_combine = pd.concat([df_existant, nouvelle_ligne], ignore_index=True)
                else:
                    df_combine = nouvelle_ligne
                
                df_combine.to_csv(fichier_data, index=False)
                st.success(f"✅ Signalement enregistré ! Total : {len(df_combine)} incidents")
                st.balloons()
                st.audio("https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3", format="audio/mp3", autoplay=True)

# ==================== PAGE : ANALYSES ====================
elif menu == "📈 Analyses":
    
    st.markdown("## 📈 Analyses approfondies")
    
    fichier_data = "data/coupures.csv"
    
    if os.path.exists(fichier_data):
        df = pd.read_csv(fichier_data)
        
        df["duree_heures"] = pd.to_numeric(df["duree_heures"], errors="coerce")
        df["impact_numerique"] = pd.to_numeric(df["impact_numerique"], errors="coerce")
        df["frequence_numerique"] = pd.to_numeric(df["frequence_numerique"], errors="coerce")
        df_clean = df.dropna(subset=["duree_heures", "impact_numerique"])
        
        tab1, tab2, tab3 = st.tabs(["📐 Régression", "🔄 ACP", "📊 Statistiques"])
        
        with tab1:
            st.markdown("### Relation durée / impact (Régression linéaire)")
            
            X = df_clean['duree_heures'].values.reshape(-1, 1)
            y = df_clean['impact_numerique'].values
            mask = ~(np.isnan(X.flatten()) | np.isnan(y))
            X_clean = X[mask].reshape(-1, 1)
            y_clean = y[mask]
            
            if len(X_clean) > 1:
                reg = LinearRegression()
                reg.fit(X_clean, y_clean)
                y_pred = reg.predict(X_clean)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X_clean, y_clean, alpha=0.6, label="Données réelles", color='#FF6B35')
                ax.plot(X_clean, y_pred, 'b-', linewidth=2, label="Tendance linéaire")
                ax.set_xlabel("Durée de la coupure (heures)")
                ax.set_ylabel("Nombre de personnes touchées")
                ax.set_title("Durée vs Impact")
                ax.legend()
                st.pyplot(fig)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Coefficient directeur", f"{reg.coef_[0]:.2f}")
                with col2:
                    st.metric("Ordonnée à l'origine", f"{reg.intercept_:.2f}")
                with col3:
                    st.metric("R² (qualité du modèle)", f"{reg.score(X_clean, y_clean):.3f}")
                
                st.caption("📌 Interprétation : Chaque heure supplémentaire de coupure touche environ {:.0f} personnes de plus".format(reg.coef_[0]))
        
        with tab2:
            st.markdown("### Projection multidimensionnelle (ACP)")
            
            features_acp = ['duree_heures', 'impact_numerique']
            if 'frequence_numerique' in df_clean.columns:
                features_acp.append('frequence_numerique')
            
            df_acp = df_clean[features_acp].dropna()
            
            if len(df_acp) > 2:
                scaler = StandardScaler()
                df_scaled = scaler.fit_transform(df_acp)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(df_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                                    c=df_acp['duree_heures'], 
                                    cmap='YlOrRd', alpha=0.7, s=80)
                ax.set_xlabel(f"Composante 1 ({pca.explained_variance_ratio_[0]:.1%})")
                ax.set_ylabel(f"Composante 2 ({pca.explained_variance_ratio_[1]:.1%})")
                ax.set_title("Visualisation ACP des incidents")
                plt.colorbar(scatter, label="Durée (heures)")
                st.pyplot(fig)
                
                st.caption(f"📊 Les 2 composantes expliquent {pca.explained_variance_ratio_.sum():.1%} de la variance totale")
        
        with tab3:
            st.markdown("### Statistiques descriptives")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Durée des coupures**")
                stats_duree = pd.DataFrame({
                    'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum', 'Q1 (25%)', 'Q3 (75%)'],
                    'Valeur (heures)': [
                        f"{df_clean['duree_heures'].mean():.1f}",
                        f"{df_clean['duree_heures'].median():.1f}",
                        f"{df_clean['duree_heures'].std():.1f}",
                        f"{df_clean['duree_heures'].min():.1f}",
                        f"{df_clean['duree_heures'].max():.1f}",
                        f"{df_clean['duree_heures'].quantile(0.25):.1f}",
                        f"{df_clean['duree_heures'].quantile(0.75):.1f}"
                    ]
                })
                st.dataframe(stats_duree, use_container_width=True)
            
            with col2:
                st.markdown("**Top 3 des zones les plus touchées**")
                top_zones = df_clean['zone'].value_counts().head(3).reset_index()
                top_zones.columns = ['Zone', 'Nombre d\'incidents']
                st.dataframe(top_zones, use_container_width=True)
        
    else:
        st.info("Aucune donnée disponible pour les analyses")

# ==================== PAGE : À PROPOS ====================
elif menu == "⚙️ À propos":
    
    st.markdown("## ⚙️ À propos")
    
    st.markdown("""
    ### 📱 Collecte Coupure BOWA
    
    **Version** : 1.0
    
    **Objectif** : Collecter et analyser les données sur les coupures d'électricité
    
    ### 🎯 Fonctionnalités
    
    - Collecte de signalements de coupures
    - Tableau de bord interactif
    - Analyses statistiques (régression, ACP)
    - Export des données
    
    ### 📧 Contact: samelphenomene@gmail.com
    ### 📞 +237 612120000
    Pour tout signalement ou suggestion, utilisez le formulaire de l'application.
    
    ---
    
    **PHENO47 - Analyse de données**
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("© 2026 Collecte Coupure BOWA | Analyse des coupures d'électricité")
