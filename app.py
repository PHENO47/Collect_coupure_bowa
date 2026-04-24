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

# ==================== LOGO ET EN-TÊTE ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-size: 4rem;'>⚡🔌⚠️</h1>
            <h1 style='color: #FF6B35;'>COLLECTE COUPURE BOWA</h1>
            <h3>Cartographie et analyse des coupures d'électricité</h3>
            <hr>
            <p style='font-size: 1.1rem;'>
            <strong>🎯 Mission :</strong> Collecter des données sur les coupures électriques pour 
            identifier les zones critiques, comprendre les causes récurrentes,<br>
            et proposer des solutions adaptées aux décideurs et aux fournisseurs d'énergie.
            </p>
            <p style='font-size: 0.9rem; color: #666;'>
            Vos données contribuent à améliorer la stabilité du réseau électrique !
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== SECTION COLLECTE ====================
with st.expander("📝 Signaler une coupure d'électricité", expanded=True):
    st.markdown("### Remplissez ce formulaire pour chaque incident")
    
    with st.form("collecte_form"):
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
            impact = st.select_slider("Impact (personnes touchées estimées)", 
                options=["Moins de 50", "50-200", "200-500", "500-2000", "Plus de 2000"])
            commentaire = st.text_area("Description / Observations supplémentaires", 
                placeholder="Ex: coupure pendant la nuit, pas de préavis, transformateur endommagé...")
        
        submitted = st.form_submit_button("✅ Enregistrer le signalement")
        
        # Mapping pour impact_numerique
        impact_map = {
            "Moins de 50": 25,
            "50-200": 125,
            "200-500": 350,
            "500-2000": 1250,
            "Plus de 2000": 3000
        }
        
        frequence_map = {
            "Première fois": 0.1,
            "Rare (1-2x/mois)": 1.5,
            "Occasionnelle (1x/semaine)": 4,
            "Fréquente (2-3x/semaine)": 10,
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
                    "duree_minutes": duree_heures * 60,
                    "impact_numerique": impact_map[impact],
                    "frequence_numerique": frequence_map[frequence]
                }])
                
                if os.path.exists(fichier_data):
                    df_existant = pd.read_csv(fichier_data)
                    df_combine = pd.concat([df_existant, nouvelle_ligne], ignore_index=True)
                else:
                    df_combine = nouvelle_ligne
                
                df_combine.to_csv(fichier_data, index=False)
                st.success(f"✅ Signalement enregistré ! Total : {len(df_combine)} incidents signalés")
                st.balloons()

st.markdown("---")

# ==================== SECTION ANALYSE ====================
st.header("📊 Analyse des données de coupures électriques")
st.markdown("*Lié aux chapitres de l'EC2 (Régression, Classification, ACP)*")

fichier_data = "data/coupures.csv"

if os.path.exists(fichier_data):
    df = pd.read_csv(fichier_data)
    
    # Forcer les types numériques
    df["duree_heures"] = pd.to_numeric(df["duree_heures"], errors="coerce")
    df["impact_numerique"] = pd.to_numeric(df["impact_numerique"], errors="coerce")
    df["frequence_numerique"] = pd.to_numeric(df["frequence_numerique"], errors="coerce")
    
    # Supprimer les lignes avec des valeurs manquantes critiques
    df_clean = df.dropna(subset=["duree_heures", "impact_numerique"])
    
    # Métriques globales
    st.subheader("📈 Indicateurs clés")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Incidents signalés", len(df_clean))
    with col2:
        st.metric("⏱️ Durée moyenne", f"{df_clean['duree_heures'].mean():.1f} h")
    with col3:
        st.metric("📊 Durée max", f"{df_clean['duree_heures'].max():.1f} h")
    with col4:
        duree_mediane = df_clean['duree_heures'].median()
        st.metric("📈 Durée médiane", f"{duree_mediane:.1f} h")
    
    st.markdown("---")
    
    # ========== RÉGRESSION LINÉAIRE SIMPLE ==========
    st.subheader("📐 Chapitre 1 - Régression linéaire simple")
    st.markdown("**Relation entre durée des coupures et impact (personnes touchées)**")
    
    X = df_clean['duree_heures'].values.reshape(-1, 1)
    y = df_clean['impact_numerique'].values
    
    # Nettoyer les NaN
    mask = ~(np.isnan(X.flatten()) | np.isnan(y))
    X_clean = X[mask].reshape(-1, 1)
    y_clean = y[mask]
    
    if len(X_clean) > 1:
        reg = LinearRegression()
        reg.fit(X_clean, y_clean)
        y_pred = reg.predict(X_clean)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Coefficient (pente)", f"{reg.coef_[0]:.2f}")
            st.metric("Ordonnée à l'origine", f"{reg.intercept_:.2f}")
            r2 = reg.score(X_clean, y_clean)
            st.metric("R² (qualité du modèle)", f"{r2:.3f}")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X_clean, y_clean, alpha=0.6, label="Données réelles")
            ax.plot(X_clean, y_pred, 'r-', linewidth=2, label="Régression linéaire")
            ax.set_xlabel("Durée de la coupure (heures)")
            ax.set_ylabel("Impact (nombre de personnes touchées)")
            ax.set_title("Régression linéaire : Durée → Impact")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("Ajoutez plus de données pour voir la régression linéaire (minimum 2 points)")
    
    st.markdown("---")
    
    # ========== RÉGRESSION MULTIPLE ==========
    st.subheader("📊 Chapitre 2 - Régression linéaire multiple")
    st.markdown("**Prédiction de l'impact selon (durée + fréquence)**")
    
    if len(df_clean) > 3 and 'frequence_numerique' in df_clean.columns:
        df_multi = df_clean.dropna(subset=['frequence_numerique'])
        X_multi = df_multi[['duree_heures', 'frequence_numerique']].values
        y_multi = df_multi['impact_numerique'].values
        
        if len(X_multi) > 2:
            reg_multi = LinearRegression()
            reg_multi.fit(X_multi, y_multi)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Coef durée", f"{reg_multi.coef_[0]:.2f}")
                st.metric("Coef fréquence", f"{reg_multi.coef_[1]:.2f}")
            
            with col2:
                st.metric("R² multiple", f"{reg_multi.score(X_multi, y_multi):.3f}")
    else:
        st.info("Ajoutez plus de données avec fréquences renseignées")
    
    st.markdown("---")
    
    # ========== ACP ==========
    st.subheader("🔄 Chapitre 3 - ACP (Analyse en Composantes Principales)")
    
    if len(df_clean) > 2:
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
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            ax.set_title("ACP : Projection des incidents de coupure")
            plt.colorbar(scatter, label="Durée (heures)")
            st.pyplot(fig)
            
            st.caption(f"Les 2 premières composantes expliquent {pca.explained_variance_ratio_.sum():.1%} de la variance")
    
    st.markdown("---")
    
    # ========== CLASSIFICATION ==========
    st.subheader("🏷️ Chapitres 4 & 5 - Classification (supervisée et non-supervisée)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cause_counts = df_clean['cause'].value_counts()
        if len(cause_counts) > 0:
            fig, ax = plt.subplots()
            cause_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_xlabel("Cause")
            ax.set_ylabel("Nombre d'incidents")
            ax.set_title("Classification par cause")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
    
    with col2:
        zone_data = df_clean.groupby('zone')['duree_heures'].agg(['count', 'mean']).sort_values('count', ascending=False)
        if len(zone_data) > 0:
            st.dataframe(zone_data.style.format({'mean': '{:.1f}'}).rename(columns={'count': 'Nb incidents', 'mean': 'Durée moyenne'}))
    
    st.markdown("---")
    
    # ========== STATISTIQUES DESCRIPTIVES ==========
    st.subheader("📋 Synthèse descriptive complète")
    
    tab1, tab2 = st.tabs(["Statistiques numériques", "Fréquences des causes"])
    
    with tab1:
        stats_df = pd.DataFrame({
            'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum', 'Q1 (25%)', 'Q3 (75%)'],
            'Durée (heures)': [
                df_clean['duree_heures'].mean(),
                df_clean['duree_heures'].median(),
                df_clean['duree_heures'].std(),
                df_clean['duree_heures'].min(),
                df_clean['duree_heures'].max(),
                df_clean['duree_heures'].quantile(0.25),
                df_clean['duree_heures'].quantile(0.75)
            ]
        })
        st.dataframe(stats_df.style.format({'Durée (heures)': '{:.1f}'}))
    
    with tab2:
        freq_causes = df_clean['cause'].value_counts()
        if len(freq_causes) > 0:
            freq_df = pd.DataFrame({
                'Cause': freq_causes.index,
                'Nombre': freq_causes.values,
                'Pourcentage (%)': (freq_causes.values / len(df_clean) * 100).round(1)
            })
            st.dataframe(freq_df)
            
            fig, ax = plt.subplots()
            ax.pie(freq_causes.values, labels=freq_causes.index, autopct='%1.1f%%')
            ax.set_title("Répartition des causes")
            st.pyplot(fig)
    
    # Téléchargement
    st.download_button(
        label="📥 Télécharger toutes les données (CSV)",
        data=df_clean.to_csv(index=False),
        file_name=f"donnees_coupures_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
else:
    st.warning("⚠️ Aucune donnée collectée pour le moment. Utilisez le formulaire ci-dessus pour signaler votre premier incident !")

st.markdown("---")
st.caption("INF232 EC2 - Analyse de données | Collecte Coupure BOWA v1.0 | Données anonymisées")