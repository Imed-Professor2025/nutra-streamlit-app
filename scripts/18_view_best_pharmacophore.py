# 18_view_best_pharmacophore.py
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
import pickle

st.title("🧬 Visualisation du meilleur pharmacophore 3D")

uploaded_file = st.file_uploader("📂 Importer le fichier best_hypothesis.pkl", type=["pkl"])
if uploaded_file is not None:
    try:
        data = pickle.load(uploaded_file)
        best_hypo = data["hypothesis"]

        st.success(f"📈 Meilleure hypothèse chargée avec {len(best_hypo)} features.")

        # Fonction de couleur
        def color_for_type(ftype):
            return {"Donor":"blue","Acceptor":"red","Aromatic":"orange","Hydrophobe":"yellow",
                    "PositiveIonizable":"purple","NegativeIonizable":"green","Ring":"cyan"}.get(ftype,"gray")

        # Fonction d'abréviation
        def abbrev_feat(f):
            mapping = {"Donor":"D","Acceptor":"A","Aromatic":"R","Hydrophobe":"H",
                       "PositiveIonizable":"P","NegativeIonizable":"N","Ring":"G"}
            return mapping.get(f["type"], "?")

        # Création du viewer
        viewer = py3Dmol.view(width=900, height=700)
        for f in best_hypo:
            pos = f["pos"]
            x, y, z = float(pos.x), float(pos.y), float(pos.z)
            c = color_for_type(f["type"])
            viewer.addSphere({'center': {'x': x, 'y': y, 'z': z}, 'radius': 1.2, 'color': c, 'alpha': 0.7})
            viewer.addLabel(abbrev_feat(f), {
                'position': {'x': x, 'y': y, 'z': z},
                'backgroundColor': c, 'fontColor': 'white', 'fontSize': 14
            })

        viewer.zoomTo()
        html_view = viewer._make_html()
        components.html(html_view, height=750, scrolling=True)

    except Exception as e:
        st.error(f"❌ Impossible de charger le fichier : {e}")
