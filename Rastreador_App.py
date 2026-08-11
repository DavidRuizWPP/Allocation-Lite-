# venv\Scripts\activate
# streamlit run Rastreador_App.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import folium
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(layout="wide", page_title="Inteligencia Territorial - Smart Fit")

# --- 1. Control de Acceso por Contraseña Corregido ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("📍 Geo Budget Allocation (Lite)")
    st.subheader("🔒 Acceso Restringido")
    
    with st.form("login_form"):
        pwd_input = st.text_input("Ingresa el Código de Acceso:", type="password")
        submit_button = st.form_submit_button("Ingresar")
        
        if submit_button:
            if pwd_input == "SmarFit.2026":
                st.session_state["authenticated"] = True
                st.success("¡Acceso concedido!")
                st.rerun()  # Recarga la app inmediatamente para mostrar el contenido
            else:
                st.error("🔑 Contraseña incorrecta. Inténtalo de nuevo.")
                
    st.stop()  # Detiene la ejecución para no mostrar nada del dashboard si no está autenticado

# --- 2. Encabezado con Logos (Smart Fit a la Izquierda y WPP a la Derecha) ---
header_col1, header_col2, header_col3 = st.columns([1, 4, 1])

with header_col1:
    try:
        st.image("Logo_SmartFit.png", width=110)
    except Exception:
        pass  # Si la imagen no se encuentra en la carpeta local, no rompe la app

with header_col2:
    st.title("📍 Inteligencia Territorial: Estrategia de Presupuesto")

with header_col3:
    try:
        st.image("Logo_WPP.png", width=150)
    except Exception:
        pass

# Matriz de Presupuesto
budget_matrix = pd.DataFrame({
    'Calentamiento': [0.25, 0.30, 0.35, 0.50, 0.60],
    'Preventa':      [0.15, 0.20, 0.25, 0.30, 0.30],
    'Apertura':      [0.60, 0.50, 0.40, 0.20, 0.10]
}, index=['Madura', 'Maturing', 'Pré-Madura', 'Ramp up', 'Sin vecino'])

with st.sidebar:
    st.header("Configuración")
    radio_km = st.slider("Radio de análisis (km)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
    
    st.header("Carga de Datos")
    file_aperturas = st.file_uploader("Subir 'UbicaciónAperturas.xlsx'", type=["xlsx"])
    file_existentes = st.file_uploader("Subir 'Analise - 469 unidades.xlsx'", type=["xlsx"])
    
    st.divider()
    if st.button("🚪 Cerrar Sesión"):
        st.session_state["authenticated"] = False
        st.rerun()

def clean_data(df):
    df.columns = df.columns.str.strip()
    df['Latitud'] = pd.to_numeric(df['Latitud'], errors='coerce')
    df['Longitud'] = pd.to_numeric(df['Longitud'], errors='coerce')
    return df.dropna(subset=['Latitud', 'Longitud'])

if file_aperturas and file_existentes:
    df_nuevas = clean_data(pd.read_excel(file_aperturas))
    df_existentes = clean_data(pd.read_excel(file_existentes))
    
    branch_names = df_nuevas['Sucursal'].tolist()
    selected_branch_name = st.sidebar.selectbox("Seleccione una sucursal:", branch_names)
    
    sucursal_info = df_nuevas[df_nuevas['Sucursal'] == selected_branch_name].iloc[0]
    
    # Lógica espacial
    coords_existentes = np.radians(df_existentes[['Latitud', 'Longitud']].values)
    tree = BallTree(coords_existentes, metric='haversine')
    
    point = np.radians([[sucursal_info['Latitud'], sucursal_info['Longitud']]])
    indices = tree.query_radius(point, r=radio_km / 6371.0)[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"Análisis: {selected_branch_name}")
        if 'Sitio Web' in sucursal_info and pd.notna(sucursal_info['Sitio Web']):
            st.link_button("Visitar Sitio Web", sucursal_info['Sitio Web'])
        
        # --- Lista de Vecinos Recuperada ---
        if len(indices) == 0:
            maturity_dist = {'Sin vecino': 1.0}
            st.warning("Sin sucursales cercanas. Se aplica matriz de 'Sin vecino'.")
        else:
            neigh_df = df_existentes.iloc[indices]
            st.write("### Sucursales Vecinas:")
            st.dataframe(neigh_df[['Sigla', 'Maturação', 'Tier']].reset_index(drop=True), use_container_width=True)
            
            maturity_dist = neigh_df['Maturação'].value_counts(normalize=True).to_dict()
            st.info(f"Se encontraron {len(indices)} sucursales cercanas.")
        
        # --- Lógica de Presupuesto ---
        recommended_budget = pd.Series([0.0, 0.0, 0.0], index=['Calentamiento', 'Preventa', 'Apertura'])
        for maturity, weight in maturity_dist.items():
            if maturity in budget_matrix.index:
                recommended_budget += budget_matrix.loc[maturity] * weight
            else:
                recommended_budget += budget_matrix.loc['Sin vecino'] * weight
        
        st.write("### Recomendación de Presupuesto (Split):")
        st.table(recommended_budget.map('{:.1%}'.format))

    with col2:
        st.subheader("Mapa")
        m = folium.Map(location=[sucursal_info['Latitud'], sucursal_info['Longitud']], zoom_start=14)
        folium.Circle(location=[sucursal_info['Latitud'], sucursal_info['Longitud']], radius=radio_km*1000, color='blue', fill=True, fill_opacity=0.1).add_to(m)
        folium.Marker([sucursal_info['Latitud'], sucursal_info['Longitud']], icon=folium.Icon(color='blue')).add_to(m)
        
        if len(indices) > 0:
            for idx in indices:
                n = df_existentes.iloc[idx]
                folium.CircleMarker([n['Latitud'], n['Longitud']], radius=5, color='red', popup=n['Sigla']).add_to(m)
        st_folium(m, width=800, height=400)

    # --- Visualizaciones Finales ---
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribución de Maduración (Vecinos)")
        fig_donut = px.pie(values=list(maturity_dist.values()), names=list(maturity_dist.keys()), hole=0.4)
        st.plotly_chart(fig_donut)
        
    with c2:
        st.subheader("Split de Presupuesto Recomendado")
        fig_bar = px.bar(x=recommended_budget.index, y=recommended_budget.values, text_auto='.1%')
        fig_bar.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_bar)

else:
    st.info("Sube los archivos en la barra lateral para generar el dashboard.")
