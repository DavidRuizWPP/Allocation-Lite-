# venv\Scripts\activate
# streamlit run LaunchAllocationEngine.py
#---------------------------------------------------------------------------
# Comienza la App
#---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, minimize, root_scalar
from scipy.stats import zscore, pearsonr
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN GENERAL Y AUTENTICACIÓN
# ==========================================
st.set_page_config(layout="wide", page_title="Budget Allocator Tool (Lite)")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("Budget Allocator Tool (Lite)")
    st.subheader("🔒 Acceso Restringido")
    
    with st.form("login_form"):
        pwd_input = st.text_input("Ingresa el Código de Acceso:", type="password")
        submit_button = st.form_submit_button("Ingresar")
        
        if submit_button:
            if pwd_input == "SmartFit.2026":
                st.session_state["authenticated"] = True
                st.success("¡Acceso concedido!")
                st.rerun()
            else:
                st.error("🔑 Contraseña incorrecta. Inténtalo de nuevo.")
                
    st.stop()

# ==========================================
# ENCABEZADO COMPARTIDO
# ==========================================
header_col1, header_col2, header_col3 = st.columns([1, 4, 1])

with header_col1:
    try:
        st.image("Logo_SmartFit.png", width=110)
    except Exception:
        pass 

with header_col2:
    st.title("Budget Allocator Tool (Lite)")
    st.markdown("*Geo-Spatial & Media Budgeting for New Openings*")

with header_col3:
    try:
        st.image("Logo_WPP.png", width=150)
    except Exception:
        pass

st.divider()

# ==========================================
# MAPA DE COLORES DE MARCA / MEDIOS (GLOBAL)
# ==========================================
color_map = {
    'Meta': '#1877F2',       # Azul corporativo Meta
    'Search': '#34A853',     # Verde Google
    'Youtube': '#FF0000',    # Rojo YouTube
    'Demand Gen': '#F3C300', # Amarillo Smart Fit
    'Display': '#9C27B0',    # Morado
    'TikTok': '#000000',     # Negro
    'Calentamiento': '#FF7F0E', # Naranja Funnel
    'Preventa': '#1F77B4',      # Azul Funnel
    'Apertura': '#2CA02C',      # Verde Funnel
    'Mínimo': '#1F77B4',        # Azul Benchmark
    'Óptimo': '#2CA02C',        # Verde Benchmark
    'Máximo': '#D62728',        # Rojo Benchmark
    'Promedio Histórico Spend': '#F3C300' # Amarillo Smart Fit Benchmark
}

# ==========================================
# 4 PESTAÑAS PRINCIPALES MAESTRAS
# ==========================================
tab_rastreador, tab_curves, tab_benchmark, tab_optimizer = st.tabs([
    "📍 Densidad de Sucursales", 
    "📊 Curvas de Respuesta", 
    "⚖️ Benchmark de Inversión", 
    "⚙️ Escenarios de Inversión"
])

# ==========================================
# PESTAÑA 1: DENSIDAD DE SUCURSALES
# ==========================================
with tab_rastreador:
    st.header("Optimización de Budget por Etapas")
    
    budget_matrix = pd.DataFrame({
        'Calentamiento': [0.25, 0.30, 0.35, 0.50, 0.60],
        'Preventa':      [0.15, 0.20, 0.25, 0.30, 0.30],
        'Apertura':      [0.60, 0.50, 0.40, 0.20, 0.10]
    }, index=['Madura', 'Maturing', 'Pré-Madura', 'Ramp up', 'Sin vecino'])
    
    st.sidebar.header("1. Carga de Datos Territoriales")
    file_aperturas = st.sidebar.file_uploader("Ubicaciones Aperturas del Periodo", type=["xlsx"], key="up_aperturas")
    file_existentes = st.sidebar.file_uploader("Ubicación Sucursales en Funcionamiento", type=["xlsx"], key="up_existentes")
    
    st.sidebar.header("1.5 Configuración Análisis Geo-Spatial")
    radio_km = st.sidebar.slider("Radio de análisis (km)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
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
        
        coords_existentes = np.radians(df_existentes[['Latitud', 'Longitud']].values)
        tree = BallTree(coords_existentes, metric='haversine')
        
        point = np.radians([[sucursal_info['Latitud'], sucursal_info['Longitud']]])
        indices = tree.query_radius(point, r=radio_km / 6371.0)[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Resultados para: {selected_branch_name}")
            if 'Sitio Web' in sucursal_info and pd.notna(sucursal_info['Sitio Web']):
                st.link_button("Visitar Sitio Web", sucursal_info['Sitio Web'])
            
            if len(indices) == 0:
                maturity_dist = {'Sin vecino': 1.0}
                st.warning("Sin sucursales cercanas. Se aplica matriz de 'Sin vecino'.")
            else:
                st.info(f"Se encontraron {len(indices)} sucursales cercanas.")

            recommended_budget = pd.Series([0.0, 0.0, 0.0], index=['Calentamiento', 'Preventa', 'Apertura'])
            
            if len(indices) > 0:
                neigh_df = df_existentes.iloc[indices]
                maturity_dist = neigh_df['Tipo de Maduración'].value_counts(normalize=True).to_dict()
            else:
                maturity_dist = {'Sin vecino': 1.0}

            for maturity, weight in maturity_dist.items():
                if maturity in budget_matrix.index:
                    recommended_budget += budget_matrix.loc[maturity] * weight
                else:
                    recommended_budget += budget_matrix.loc['Sin vecino'] * weight
            
            if len(indices) > 0:
                st.write("### Sucursales Vecinas:")
                st.dataframe(
                    neigh_df[['Sigla', 'Tipo de Maduración', 'Tier']].reset_index(drop=True), 
                    use_container_width=True,
                    height=200
                )

        with col2:
            st.subheader("Geolocalizador Sucursales Smart Fit")
            m = folium.Map(location=[sucursal_info['Latitud'], sucursal_info['Longitud']], zoom_start=14)
            folium.Circle(location=[sucursal_info['Latitud'], sucursal_info['Longitud']], radius=radio_km*1000, color='blue', fill=True, fill_opacity=0.1).add_to(m)
            folium.Marker([sucursal_info['Latitud'], sucursal_info['Longitud']], icon=folium.Icon(color='blue')).add_to(m)
            
            if len(indices) > 0:
                for idx in indices:
                    n = df_existentes.iloc[idx]
                    folium.CircleMarker([n['Latitud'], n['Longitud']], radius=5, color='red', popup=n['Sigla']).add_to(m)
            st_folium(m, width=800, height=400)

        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Distribución de Maduración (Vecinos)")
            fig_donut = px.pie(
                values=list(maturity_dist.values()), 
                names=list(maturity_dist.keys()), 
                hole=0.4
            )
            fig_donut.update_traces(textinfo='percent+label', textfont_size=15)
            fig_donut.update_layout(legend=dict(font=dict(size=14)))
            st.plotly_chart(fig_donut, theme="streamlit")
            
        with c2:
            st.subheader("SOI por etapa de Funnel")
            fig_bar = px.bar(
                x=recommended_budget.index, 
                y=recommended_budget.values, 
                text=recommended_budget.values,
                color=recommended_budget.index,
                color_discrete_map=color_map
            )
            fig_bar.update_traces(texttemplate='%{text:.1%}', textposition='outside', textfont_size=15)
            fig_bar.update_layout(
                xaxis_title=None,
                yaxis_title=None,
                xaxis=dict(tickfont=dict(size=14)),
                yaxis=dict(tickformat='.0%', tickfont=dict(size=14)),
                showlegend=False
            )
            fig_bar.update_yaxes(range=[0, 1.15])
            st.plotly_chart(fig_bar, theme="streamlit")

        # ==========================================
        # SECCIÓN NUEVA: BASELINE SPEND (REGRESIÓN LINEAL)
        # ==========================================
        st.divider()
        st.header("📊 Baseline de Inversión para Apertura")
        st.markdown("Estimación de inversión base basada en modelo de regresión lineal múltiple con los parámetros espaciales actuales.")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            poblacion_input = st.number_input(
                "Población alrededor (dentro del radio seleccionado):", 
                min_value=0.0, 
                value=50000.0, 
                step=5000.0,
                key=f"pob_{selected_branch_name}"
            )
        with col_b2:
            num_vecinos = len(indices)
            st.metric(label="Número de Vecinos (en el radio)", value=num_vecinos)
            
        baseline_spend = 9006.42 + (0.0031 * poblacion_input) + (63.6 * num_vecinos)
        st.markdown(f"### 💡 El Budget Sufficiency para **{selected_branch_name}** es : **${max(baseline_spend, 0.0):,.2f}** por mes")

    else:
        st.info("Sube los archivos de ubicaciones de Smart Fit en la barra lateral para habilitar el análisis territorial.")


# ==========================================
# FUNCIONES MATEMÁTICAS MMM
# ==========================================
def hill(x, C, alpha, beta, scale):
    xs = x / scale
    bs = beta / scale
    return C * (xs**alpha) / (bs**alpha + xs**alpha + 1e-10)

def hill_derivative(x, C, alpha, beta, scale):
    xs = x / scale
    bs = beta / scale
    num = C * alpha * (bs**alpha) * (xs**(alpha - 1))
    den = scale * ((bs**alpha + xs**alpha)**2)
    return num / den

def fit_curves(df, plat_col, spend_col, kpi_col, remove_outliers=True):
    plataformas = df[plat_col].unique()
    params = {}
    for plat in plataformas:
        sub = df[df[plat_col] == plat].sort_values(spend_col).copy()
        if remove_outliers and len(sub) > 5:
            z_s = np.abs(zscore(sub[spend_col]))
            z_k = np.abs(zscore(sub[kpi_col]))
            sub_clean = sub[(z_s < 3.0) & (z_k < 3.0)]
        else:
            sub_clean = sub
            
        x = sub_clean[spend_col].values
        y = sub_clean[kpi_col].values
        if len(x) < 3 or y.max() == 0: continue
            
        scale = x.max()
        c_min = y.max()
        c_max = y.max() * 2.0
        p0 = [c_min * 1.2, 2.0, np.median(x)]
        bounds = ([c_min, 1.1, x.min() * 0.1], [c_max, 10.0, x.max() * 2.0])
        
        try:
            popt, _ = curve_fit(lambda x_val, C, a, b: hill(x_val, C, a, b, scale), x, y, p0=p0, bounds=bounds, maxfev=10000)
            C, alpha, beta = popt
            
            y_fit = hill(x, C, alpha, beta, scale)
            corr, _ = pearsonr(y, y_fit) if np.std(y) > 0 and np.std(y_fit) > 0 else (np.nan, np.nan)
            
            max_xs = (beta / scale) * (0.85 / 0.15)**(1 / alpha)
            max_inv = max_xs * scale
            
            if alpha > 1:
                inflection_xs = (beta / scale) * ((alpha - 1) / (alpha + 1))**(1 / alpha)
                min_inv = inflection_xs * scale
            else:
                min_inv = 0
                
            params[plat] = {
                'C': C, 'alpha': alpha, 'beta': beta, 'scale': scale, 
                'corr': corr, 'min_inv': min_inv, 'max_inv': max_inv,
                'x_raw': x, 'y_raw': y
            }
        except:
            pass
    return params


# Configuración de carga para datos de medios en sidebar
st.sidebar.divider()
st.sidebar.header("2. Actividad Histórica por Plataforma")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=['csv'], key="up_medios")

if uploaded_file is not None:
    try:
        df_medios = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        df_medios = pd.read_csv(uploaded_file, encoding='latin1')
        
    plat_col = st.sidebar.selectbox("Nivel de Optimización", df_medios.columns, index=0)
    spend_col = st.sidebar.selectbox("Métrica de Media", df_medios.columns, index=1)
    kpi_col = st.sidebar.selectbox("Métrica de Negocio Objetivo", df_medios.columns, index=2)
    
    with st.spinner("Calculando modelos matemáticos..."):
        params_clean = fit_curves(df_medios, plat_col, spend_col, kpi_col, remove_outliers=True)
        params_all = fit_curves(df_medios, plat_col, spend_col, kpi_col, remove_outliers=False)
else:
    df_medios = None
    params_clean, params_all = {}, {}
    plat_col, spend_col, kpi_col = None, None, None


# ==========================================
# PESTAÑA 2: RESPONSE CURVES FIT
# ==========================================
with tab_curves:
    st.header("Construcción de Curvas de Respuesta")
    if uploaded_file is not None and params_clean:
        
        st.subheader("Parámetros")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Modelo Conservador**")
            res_clean = [{'Plataforma': k, 'Límite de Conversiones': v['C'], 'Celeridad de Eficiencia': v['alpha'], 'Escala de Inversión': v['beta']} for k, v in params_clean.items()]
            st.dataframe(pd.DataFrame(res_clean).style.format({'C': "{:.2f}", 'Alpha': "{:.4f}", 'Beta': "{:.2f}"}))
        with col_t2:
            st.markdown("**Modelo Agresivo (Considera resultados sobresalientes)**")
            res_all = [{'Plataforma': k, 'Límite de Conversiones': v['C'], 'Celeridad de Eficiencia': v['alpha'], 'Escala de Inversión': v['beta']} for k, v in params_all.items()]
            st.dataframe(pd.DataFrame(res_all).style.format({'C': "{:.2f}", 'Alpha': "{:.4f}", 'Beta': "{:.2f}"}))

        st.divider()
        st.subheader("Curvas Ajustadas")
        
        all_plats = list(set(list(params_clean.keys()) + list(params_all.keys())))
        
        for plat in all_plats:
            st.markdown(f"{plat}")
            col_left, col_right = st.columns(2)
            
            with col_left:
                if plat in params_clean:
                    p = params_clean[plat]
                    x_line = np.linspace(p['x_raw'].min(), p['max_inv'], 300)
                    y_line = hill(x_line, p['C'], p['alpha'], p['beta'], p['scale'])
                    roi_line = y_line / x_line
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=p['x_raw'], y=p['y_raw'], mode='markers', name='Resultados', marker=dict(color='green', size=8)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Curva de Respuesta', line=dict(color='#1f77b4', width=3)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=x_line, y=roi_line, mode='lines', name='Eficiencia Marginal', line=dict(color='#d62728', width=2, dash='dash')), secondary_y=True)
                    
                    fig.update_layout(title=f"(Coef. Correlación = {p['corr']:.2f})", hovermode="x unified", height=350, margin=dict(l=20, r=20, t=40, b=20))
                    fig.update_yaxes(title_text=kpi_col, secondary_y=False)
                    fig.update_yaxes(title_text="Eficiencia Marginal", secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                else:
                    st.info("Sin datos suficientes en este modelo.")

            with col_right:
                if plat in params_all:
                    p = params_all[plat]
                    x_line = np.linspace(p['x_raw'].min(), p['max_inv'], 300)
                    y_line = hill(x_line, p['C'], p['alpha'], p['beta'], p['scale'])
                    roi_line = y_line / x_line
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=p['x_raw'], y=p['y_raw'], mode='markers', name='Resultados', marker=dict(color='orange', size=8)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Curva de Respuesta', line=dict(color='#ff7f0e', width=3)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=x_line, y=roi_line, mode='lines', name='Eficiencia Marginal', line=dict(color='#d62728', width=2, dash='dash')), secondary_y=True)
                    
                    fig.update_layout(title=f"(Coef. Correlación = {p['corr']:.2f})", hovermode="x unified", height=350, margin=dict(l=20, r=20, t=40, b=20))
                    fig.update_yaxes(title_text=kpi_col, secondary_y=False)
                    fig.update_yaxes(title_text="Eficiencia Marginal", secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                else:
                    st.info("Sin datos suficientes en este modelo.")
            st.divider()
    else:
        st.info("👈 Sube tu archivo CSV de inversión en la barra lateral para visualizar las curvas de respuesta y hiperparámetros.")


# ==========================================
# PESTAÑA 3: BUDGET BENCHMARK
# ==========================================
with tab_benchmark:
    st.header("Budget Benchmark")
    if uploaded_file is not None and params_clean:
        
        historical_cacs = {}
        avg_spends = {}
        for plat in params_clean.keys():
            sub_plat = df_medios[df_medios[plat_col] == plat]
            total_spend = sub_plat[spend_col].sum()
            total_kpi = sub_plat[kpi_col].sum()
            avg_spend = sub_plat[spend_col].mean()
            
            avg_spends[plat] = avg_spend if not np.isnan(avg_spend) else 0.0
            if total_kpi > 0:
                historical_cacs[plat] = total_spend / total_kpi
            else:
                historical_cacs[plat] = 300.0

        st.markdown("Ingresa el Costo por Adquisición Objetivo por plataforma (Predeterminado está el promedio histórico):")
        
        target_cacs = {}
        cols = st.columns(len(params_clean))
        for idx, plat in enumerate(params_clean.keys()):
            with cols[idx]:
                default_cac = float(historical_cacs.get(plat, 300.0))
                target_cacs[plat] = st.number_input(f"CAC para {plat}", value=default_cac, step=10.0, key=f"cac_{plat}")
                
        if st.button("Calcular Benchmark"):
            bench_results = []
            for plat, p in params_clean.items():
                min_inv = p['min_inv']
                max_inv = p['max_inv']
                
                def marginal_eq(val):
                    return hill_derivative(val, p['C'], p['alpha'], p['beta'], p['scale']) - (1 / target_cacs[plat])
                
                try:
                    res = root_scalar(marginal_eq, bracket=[min_inv + 1, p['scale'] * 20], method='brentq')
                    opt_inv = res.root
                except:
                    opt_inv = np.nan
                    
                bench_results.append({
                    'Plataforma': plat,
                    'Mínimo': min_inv,
                    'Óptimo': opt_inv,
                    'Máximo': max_inv
                })
                
            df_bench = pd.DataFrame(bench_results)
            st.dataframe(df_bench.style.format({
                'Mínimo': "${:,.2f}",
                'Óptimo': "${:,.2f}",
                'Máximo': "${:,.2f}"
            }, na_rep="No Alcanzado (Muy Caro)"))

            st.divider()
            st.subheader("📈 Comparativa de Inversión vs. Promedio Histórico por Plataforma")
            
            for row in bench_results:
                plat = row['Plataforma']
                opt_val = row['Óptimo']
                opt_plot = opt_val if not np.isnan(opt_val) else 0.0
                
                metrics_df = pd.DataFrame({
                    'Métrica': ['Mínimo', 'Óptimo', 'Máximo', 'Promedio Histórico Spend'],
                    'Inversión ($)': [
                        row['Mínimo'], 
                        opt_plot, 
                        row['Máximo'], 
                        avg_spends.get(plat, 0.0)
                    ]
                })
                
                fig_bench = px.bar(
                    metrics_df, 
                    x='Métrica', 
                    y='Inversión ($)', 
                    text='Inversión ($)',
                    color='Métrica',
                    color_discrete_map=color_map,
                    title=f"Estructura de Inversión - {plat}"
                )
                fig_bench.update_traces(texttemplate='$%{text:,.2f}', textposition='outside', textfont_size=15)
                fig_bench.update_layout(
                    xaxis_title=None,
                    yaxis_title="Inversión ($)",
                    xaxis=dict(tickfont=dict(size=14)),
                    yaxis=dict(tickformat='$,.0f', tickfont=dict(size=14), title_font=dict(size=15)),
                    showlegend=False
                )
                st.plotly_chart(fig_bench, use_container_width=True, theme="streamlit")

    else:
        st.info("👈 Sube tu archivo CSV de inversión en la barra lateral para calcular el benchmark de presupuestos.")


# ==========================================
# PESTAÑA 4: ALLOCATOR OPTIMIZER
# ==========================================
with tab_optimizer:
    st.header("Allocator Optimizer")
    
    if uploaded_file is not None and params_clean:
        escenario = st.radio("Selecciona el tipo de optimización:", [
            "1. Presupuesto Libre (Maximizar KPI)", 
            "2. Presupuestos Mínimos y Máximos", 
            "3. Búsqueda de Objetivo"
        ])
        
        def render_optimizer_charts(df_res, title_suffix=""):
            st.divider()
            col_g1, col_g2 = st.columns(2)
            
            kpi_col_name = [c for c in df_res.columns if 'Proyectada' in c or 'Aportados' in c][0]
            total_kpi = df_res[kpi_col_name].sum()
            
            if total_kpi > 0:
                df_res['Share_Conv'] = (df_res[kpi_col_name] / total_kpi) * 100
            else:
                df_res['Share_Conv'] = 0.0

            with col_g1:
                st.markdown(f"### Optimal Digital Mix {title_suffix}")
                fig_donut = px.pie(
                    df_res, 
                    values='Inversión Óptima' if 'Inversión Óptima' in df_res.columns else 'Inversión Requerida', 
                    names='Plataforma', 
                    hole=0.4,
                    color='Plataforma',
                    color_discrete_map=color_map
                )
                fig_donut.update_traces(textinfo='percent+label', textfont_size=15)
                fig_donut.update_layout(legend=dict(font=dict(size=14)))
                st.plotly_chart(fig_donut, use_container_width=True, theme="streamlit")
                
            with col_g2:
                st.markdown(f"### Share of Conversion por Medio {title_suffix}")
                fig_bar_kpi = px.bar(
                    df_res, 
                    x='Plataforma', 
                    y='Share_Conv', 
                    text='Share_Conv',
                    color='Plataforma',
                    color_discrete_map=color_map
                )
                fig_bar_kpi.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont_size=15)
                fig_bar_kpi.update_layout(
                    xaxis_title=None, 
                    yaxis_title="Share of Conversion (%)",
                    xaxis=dict(tickfont=dict(size=14)),
                    yaxis=dict(tickfont=dict(size=14), title_font=dict(size=15)),
                    showlegend=False
                )
                fig_bar_kpi.update_yaxes(range=[0, max(df_res['Share_Conv'].max() * 1.15, 10.0)])
                st.plotly_chart(fig_bar_kpi, use_container_width=True, theme="streamlit")

        if escenario == "1. Presupuesto Libre (Maximizar KPI)":
            st.subheader("Maximiza resultados dado un presupuesto usando Touchpoints Eficientes.")
            budget_free = st.number_input("Presupuesto Total Disponible", value=100000.0, step=5000.0)
            
            if st.button("Optimizar (Libre)"):
                plats_opt = list(params_clean.keys())
                
                def obj_free(shares): 
                    return -np.sum([hill(shares[i] * budget_free, params_clean[p]['C'], params_clean[p]['alpha'], params_clean[p]['beta'], params_clean[p]['scale']) for i, p in enumerate(plats_opt)])
                
                def cons_free(shares): 
                    return 1.0 - np.sum(shares)
                
                res = minimize(obj_free, [1.0/len(plats_opt)]*len(plats_opt), method='SLSQP', bounds=[(0.0, 1.0) for _ in plats_opt], constraints=[{'type': 'eq', 'fun': cons_free}])
                
                if res.success:
                    res_df = []
                    for i, plat in enumerate(plats_opt):
                        inv = (res.x[i] * budget_free) if (res.x[i] * budget_free) >= 1 else 0.0
                        conv = hill(inv, params_clean[plat]['C'], params_clean[plat]['alpha'], params_clean[plat]['beta'], params_clean[plat]['scale']) if inv > 0 else 0
                        res_df.append({'Plataforma': plat, 'Inversión Óptima': inv, f'{kpi_col} Proyectada': conv})
                    
                    df_result_free = pd.DataFrame(res_df)
                    st.success(f"Optimización Exitosa. {kpi_col} Totales: {df_result_free[f'{kpi_col} Proyectada'].sum():,.1f}")
                    st.dataframe(df_result_free.style.format({'Inversión Óptima': "${:,.2f}", f'{kpi_col} Proyectada': "{:,.1f}"}))
                    
                    render_optimizer_charts(df_result_free)

        elif escenario == "2. Presupuestos Mínimos y Máximos":
            st.subheader("Maximiza resultados asegurando una inversión mínima y un tope máximo por Touchpoint.")
            budget_cons = st.number_input("Presupuesto Total Disponible", value=100000.0, step=5000.0, key="budget_cons")
            
            st.markdown("Ingresa la inversión **Mínima** y **Máxima** por plataforma:")
            min_spends = {}
            max_spends = {}
            
            # Creamos columnas dobles para cada plataforma (Mínimo y Máximo)
            cols_cons = st.columns(len(params_clean))
            for idx, plat in enumerate(params_clean.keys()):
                with cols_cons[idx]:
                    st.markdown(f"**{plat}**")
                    min_spends[plat] = st.number_input(f"Mínimo {plat}", value=1000.0, step=500.0, key=f"min_{plat}")
                    max_spends[plat] = st.number_input(f"Máximo {plat}", value=50000.0, step=5000.0, key=f"max_{plat}")
            
            if st.button("Optimizar con Restricciones (Min/Max)"):
                suma_minimos = sum(min_spends.values())
                suma_maximos = sum(max_spends.values())
                
                if suma_minimos > budget_cons:
                    st.error(f"⚠️ La suma de los mínimos (${suma_minimos:,.2f}) es mayor al presupuesto total (${budget_cons:,.2f}).")
                elif suma_maximos < budget_cons:
                    st.error(f"⚠️ La suma de los máximos (${suma_maximos:,.2f}) es menor al presupuesto total (${budget_cons:,.2f}). Es imposible repartir el dinero.")
                else:
                    plats_opt = list(params_clean.keys())
                    
                    def obj_cons(shares): 
                        return -np.sum([hill(shares[i] * budget_cons, params_clean[p]['C'], params_clean[p]['alpha'], params_clean[p]['beta'], params_clean[p]['scale']) for i, p in enumerate(plats_opt)])
                    
                    def cons_cons(shares): 
                        return 1.0 - np.sum(shares)
                    
                    # Definición dinámica de bounds basados en los mínimos y máximos en dólares convertidos a shares (0 a 1)
                    bounds = [(min_spends[p] / budget_cons, max_spends[p] / budget_cons) for p in plats_opt]
                    
                    presupuesto_sobrante = budget_cons - suma_minimos
                    x0 = [(min_spends[p] + (presupuesto_sobrante / len(plats_opt))) / budget_cons for p in plats_opt]
                    
                    res = minimize(obj_cons, x0, method='SLSQP', bounds=bounds, constraints=[{'type': 'eq', 'fun': cons_cons}])
                    
                    if res.success:
                        res_df = []
                        for i, plat in enumerate(plats_opt):
                            inv = (res.x[i] * budget_cons) if (res.x[i] * budget_cons) >= 1 else 0.0
                            conv = hill(inv, params_clean[plat]['C'], params_clean[plat]['alpha'], params_clean[plat]['beta'], params_clean[plat]['scale']) if inv > 0 else 0
                            
                            min_req = min_spends[plat]
                            max_req = max_spends[plat]
                            
                            if abs(inv - min_req) < 10:
                                nota = "🔒 Mínimo Forzado"
                            elif abs(inv - max_req) < 10:
                                nota = "🔒 Máximo Forzado"
                            else:
                                nota = "📈 Optimizado"
                            
                            res_df.append({
                                'Plataforma': plat, 
                                'Inversión Óptima': inv, 
                                f'{kpi_col} Proyectada': conv,
                                'Estatus': nota
                            })
                            
                        df_result_cons = pd.DataFrame(res_df)
                        st.success(f"Optimización Exitosa. {kpi_col} Totales: {df_result_cons[f'{kpi_col} Proyectada'].sum():,.1f}")
                        st.dataframe(df_result_cons.style.format({'Inversión Óptima': "${:,.2f}", f'{kpi_col} Proyectada': "{:,.1f}"}))
                        
                        render_optimizer_charts(df_result_cons)
                    else:
                        st.error("❌ El optimizador no pudo encontrar una solución con los límites min/max proporcionados. Revisa que el presupuesto total quepa dentro de las restricciones.")

        elif escenario == "3. Búsqueda de Objetivo":
            st.subheader("Encuentra la ruta más barata para llegar a una meta, comparando escenarios con y sin Outliers.")
            target_kpi = st.number_input(f"Objetivo de {kpi_col} a alcanzar", value=400.0, step=50.0)
            
            if st.button("Calcular Ruta Óptima"):
                plats_opt = list(params_clean.keys())
                
                def run_goal_seek(params_dict, name):
                    max_techo = sum([params_dict[p]['C'] for p in plats_opt]) * 0.85
                    if target_kpi > max_techo:
                        return None, max_techo
                    
                    ESCALA = 100000.0 
                    
                    def obj_gs(inv_scaled): return np.sum(inv_scaled)
                    def cons_gs(inv_scaled): 
                        return sum([hill(inv_scaled[i] * ESCALA, params_dict[p]['C'], params_dict[p]['alpha'], params_dict[p]['beta'], params_dict[p]['scale']) for i, p in enumerate(plats_opt)]) - target_kpi
                    
                    res = minimize(obj_gs, [(15000.0 / ESCALA)]*len(plats_opt), method='SLSQP', bounds=[(0.0, np.inf)]*len(plats_opt), constraints=[{'type': 'eq', 'fun': cons_gs}])
                    
                    if res.success:
                        df_res = []
                        for i, plat in enumerate(plats_opt):
                            inv = (res.x[i] * ESCALA) if (res.x[i] * ESCALA) >= 1 else 0.0
                            conv = hill(inv, params_dict[plat]['C'], params_dict[plat]['alpha'], params_dict[plat]['beta'], params_dict[plat]['scale']) if inv > 0 else 0
                            df_res.append({'Plataforma': plat, 'Inversión Requerida': inv, f'{kpi_col} Proyectada': conv})
                        return pd.DataFrame(df_res), max_techo
                    return pd.DataFrame(), max_techo
                
                df_clean, max_c = run_goal_seek(params_clean, "Conservador")
                df_all, max_a = run_goal_seek(params_all, "Agresivo")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("### Escenario 1: Conservador")
                    if df_clean is None:
                        st.warning(f"Objetivo inalcanzable. Techo práctico: {max_c:,.0f}")
                    elif df_clean.empty:
                        st.error("No convergió.")
                    else:
                        st.success(f"Presupuesto Total Requerido: ${df_clean['Inversión Requerida'].sum():,.2f}")
                        st.dataframe(df_clean.style.format({'Inversión Requerida': "${:,.2f}", f'{kpi_col} Proyectada': "{:,.1f}"}))
                        render_optimizer_charts(df_clean, "(Conservador)")
                        
                with col_b:
                    st.markdown("### Escenario 2: Agresivo")
                    if df_all is None:
                        st.warning(f"Objetivo inalcanzable. Techo práctico: {max_a:,.0f}")
                    elif df_all.empty:
                        st.error("No convergió.")
                    else:
                        st.success(f"Presupuesto Total Requerido: ${df_all['Inversión Requerida'].sum():,.2f}")
                        st.dataframe(df_all.style.format({'Inversión Requerida': "${:,.2f}", f'{kpi_col} Proyectada': "{:,.1f}"}))
                        render_optimizer_charts(df_all, "(Agresivo)")
    else:
        st.info("👈 Sube tu archivo CSV de inversión en la barra lateral para activar las opciones de optimizador.")

# ==========================================
# CERRAR SESIÓN
# ==========================================
st.sidebar.divider()
if st.sidebar.button("🚪 Cerrar Sesión"):
    st.session_state["authenticated"] = False
    st.rerun()
