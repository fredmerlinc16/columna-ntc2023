import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    from fpdf import FPDF
    FPDF_DISPONIBLE = True
except ImportError:
    FPDF_DISPONIBLE = False

st.set_page_config(page_title="Diseño Integral de Columnas NTC 2023", layout="wide")

# --- CATÁLOGO MAESTRO DE ACERO ---
CATALOGO_VARILLAS = {
    "#2.5 (8mm)": {"area": 0.49, "peso": 0.395, "diam": 0.79},
    "#3 (3/8\")": {"area": 0.71, "peso": 0.557, "diam": 0.95},
    "#4 (1/2\")": {"area": 1.27, "peso": 0.994, "diam": 1.27},
    "#5 (5/8\")": {"area": 1.99, "peso": 1.560, "diam": 1.59},
    "#6 (3/4\")": {"area": 2.85, "peso": 2.235, "diam": 1.91},
    "#8 (1\")":   {"area": 5.07, "peso": 3.975, "diam": 2.54},
    "#10 (1 1/4\")":{"area": 7.92, "peso": 6.225, "diam": 3.18},
    "#12 (1 1/2\")":{"area": 11.40, "peso": 8.938, "diam": 3.81}
}

st.title("Diseño y Revisión Integral de Columnas (NTC 2023)")
st.markdown("Revisión Biaxial 3D, Cortante, Derivas, Detallado BIM 2D/3D y **Generador de Memorias PDF**.")

# --- BARRA LATERAL ---
st.sidebar.header("1. Materiales")
fc = st.sidebar.number_input("f'c [kg/cm²]", 150, 800, 250, 10)
fy = st.sidebar.number_input("fy [kg/cm²]", 2000, 6000, 4200, 100)
FR_flex = st.sidebar.number_input("FR (Flexocompresión)", 0.5, 0.9, 0.65)
FR_cort = st.sidebar.number_input("FR (Cortante)", 0.5, 0.9, 0.75)

st.sidebar.header("2. Geometría y Refuerzo")
forma = st.sidebar.selectbox("Forma de la sección", ["Rectangular/Cuadrada", "Circular"])

if forma == "Rectangular/Cuadrada":
    b = st.sidebar.number_input("Base, b (Eje X) [cm]", 20, 200, 40, 5)
    h = st.sidebar.number_input("Peralte, h (Eje Y) [cm]", 20, 200, 50, 5)
    Ag = b * h
else:
    D = st.sidebar.number_input("Diámetro, D [cm]", 20, 200, 50, 5)
    b, h = D, D
    Ag = np.pi * (D**2) / 4

rec = st.sidebar.number_input("Recubrimiento al centroide [cm]", 3, 15, 5, 1)
As = st.sidebar.number_input("Área total de acero de cálculo, As [cm²]", 1.0, 300.0, 20.0, 1.0)

cuantia = (As / Ag) * 100

st.sidebar.header("3. Estribos (Cortante)")
calibres_estribos = ["#2.5 (8mm)", "#3 (3/8\")", "#4 (1/2\")", "#5 (5/8\")"]
tipo_estribo = st.sidebar.selectbox("Calibre del estribo", calibres_estribos, index=1)
a_estribo = CATALOGO_VARILLAS[tipo_estribo]["area"]
ramas_x = st.sidebar.number_input("Ramas cortando Vx", 2, 10, 3)
ramas_y = st.sidebar.number_input("Ramas cortando Vy", 2, 10, 3)
separacion_s = st.sidebar.number_input("Separación propuesta, s [cm]", 5, 50, 15)

st.sidebar.header("4. Esbeltez")
Lu = st.sidebar.number_input("Altura libre de columna, Lu [m]", 1.0, 12.0, 3.0, 0.1)
k_factor = st.sidebar.number_input("Factor de longitud efectiva, k", 0.5, 3.0, 1.0, 0.1)
deriva_max_permisible = st.sidebar.selectbox("Deriva máxima permisible", [0.015, 0.012, 0.010, 0.005], format_func=lambda x: f"{x*100}%")

# --- MOTOR DE CÁLCULO ---
def calcular_curva_uniaxial(base, peralte, recub, area_acero, f_c, f_y):
    f_bic = 0.85 * f_c
    beta1 = 0.85 if f_c <= 280 else max(0.65, 0.85 - 0.004 * (f_c - 280))
    Es = 2000000 
    d, d_prime, As_cara = peralte - recub, recub, area_acero / 2.0 
    P_vals, M_vals = [], []
    for c in np.linspace(0.01, peralte * 1.5, 45):
        a = min(beta1 * c, peralte)
        Cc = f_bic * a * base
        fs1 = min(f_y, max(-f_y, (0.003 * (c - d_prime) / c) * Es))
        fs2 = min(f_y, max(-f_y, (0.003 * (d - c) / c) * Es))
        Pn = Cc + (As_cara * fs1) - (As_cara * fs2)
        Mn = Cc*(peralte/2 - a/2) + (As_cara*fs1)*(peralte/2 - d_prime) + (As_cara*fs2)*(d - peralte/2)
        P_vals.append(Pn / 1000)
        M_vals.append(abs(Mn / 100000))
    Po = (f_bic * (base*peralte - area_acero) + f_y * area_acero) / 1000
    To = -(f_y * area_acero) / 1000
    return np.array([To] + sorted(P_vals) + [Po]), np.array([0.0] + [M_vals[i] for i in np.argsort(P_vals)] + [0.0])

def capacidad_cortante(b_web, d_eff, f_c, f_y, a_v, s_sep, fr_v):
    f_star_c = 0.8 * f_c
    Vcr = fr_v * 0.5 * np.sqrt(f_star_c) * b_web * d_eff / 1000 
    Vsr = fr_v * (a_v * f_y * d_eff) / s_sep / 1000             
    Vr_max = fr_v * 2.0 * np.sqrt(f_star_c) * b_web * d_eff / 1000
    return Vcr, Vsr, min(Vcr + Vsr, Vr_max), Vr_max

P_x, M_x = calcular_curva_uniaxial(b, h, rec, As, fc, fy) 
P_y, M_y = calcular_curva_uniaxial(h, b, rec, As, fc, fy) 
P_z = np.linspace(max(P_x[0], P_y[0]), min(P_x[-1], P_y[-1]), 40)
Mx_interp, My_interp = np.interp(P_z, P_x, M_x) * FR_flex, np.interp(P_z, P_y, M_y) * FR_flex
P_z_diseno = P_z * FR_flex

theta = np.linspace(0, 2*np.pi, 40)
THETA, P_GRID = np.meshgrid(theta, P_z_diseno)
MX_GRID, MY_GRID = Mx_interp[:, np.newaxis] * np.cos(THETA), My_interp[:, np.newaxis] * np.sin(THETA)

d_x, bw_x = (b - rec if forma == "Rectangular/Cuadrada" else 0.8 * D), (h if forma == "Rectangular/Cuadrada" else D)
Vcr_x, Vsr_x, VR_x, VRmax_x = capacidad_cortante(bw_x, d_x, fc, fy, a_estribo * ramas_x, separacion_s, FR_cort)

d_y, bw_y = (h - rec if forma == "Rectangular/Cuadrada" else 0.8 * D), (b if forma == "Rectangular/Cuadrada" else D)
Vcr_y, Vsr_y, VR_y, VRmax_y = capacidad_cortante(bw_y, d_y, fc, fy, a_estribo * ramas_y, separacion_s, FR_cort)

# Cálculo de estribos colocados
dim_min, dim_max, H_cm = min(b, h), max(b, h), Lu * 100.0
Lo = max(H_cm / 6.0, dim_max, 45.0)
toda_confinada = False
if 2 * Lo >= H_cm: Lo, toda_confinada = H_cm / 2.0, True

posiciones_y = []
y_actual = separacion_s / 2.0
while y_actual <= Lo:
    posiciones_y.append(y_actual)
    y_actual += separacion_s
if not toda_confinada:
    y_actual = Lo + separacion_s
    while y_actual < (H_cm - Lo):
        posiciones_y.append(y_actual)
        y_actual += separacion_s
y_actual = H_cm - Lo + (separacion_s / 2.0 if toda_confinada else 0)
while y_actual < H_cm:
    if y_actual not in posiciones_y: posiciones_y.append(y_actual)
    y_actual += separacion_s
num_estribos_total = len(posiciones_y)

# --- TABLA ETABS ---
st.header("5. Solicitaciones de Diseño (ETABS)")
df_etabs = pd.DataFrame({
    "Combo": ["Sismo_X", "Sismo_Y"], "Pu [Ton]": [120.0, 115.0],
    "Mux [Ton-m]": [18.5, 4.2], "Muy [Ton-m]": [3.1, 16.8],
    "Vux [Ton]": [11.2, 1.5], "Vuy [Ton]": [1.8, 13.4],
    "Δx_top [cm]": [3.2, 0.4], "Δy_top [cm]": [0.5, 3.8]
})
edit_df = st.data_editor(df_etabs, num_rows="dynamic", use_container_width=True)

res_globales = []
for idx, row in edit_df.iterrows():
    p, mx, my = row["Pu [Ton]"], abs(row["Mux [Ton-m]"]), abs(row["Muy [Ton-m]"])
    vx, vy, dx, dy = abs(row["Vux [Ton]"]), abs(row["Vuy [Ton]"]), abs(row["Δx_top [cm]"]), abs(row["Δy_top [cm]"])
    if p > P_z_diseno.max() or p < P_z_diseno.min(): dc_flex = 999.0
    else:
        mx_c, my_c = np.interp(p, P_z_diseno, Mx_interp), np.interp(p, P_z_diseno, My_interp)
        dc_flex = np.sqrt((mx/mx_c)**2 + (my/my_c)**2) if (mx_c > 0 and my_c > 0) else 999.0
    dc_cortante = max(vx / VR_x, vy / VR_y)
    deriva_max = max((dx / (Lu * 100)), (dy / (Lu * 100)))
    estatus = "✅ APROBADO" if (dc_flex <= 1.0 and dc_cortante <= 1.0 and deriva_max <= deriva_max_permisible) else "❌ RECHAZADO"
    res_globales.append({"Combo": row["Combo"], "D/C Flexión": round(dc_flex, 2), "D/C Cortante": round(dc_cortante, 2), "Deriva Máx": f"{deriva_max*100:.2f}%", "Estatus": estatus})

df_master = pd.DataFrame(res_globales)

# --- PESTAÑAS DE VISUALIZACIÓN ---
st.divider()
t_res, t_3d, t_cort, t_det, t_gen = st.tabs([
    "📋 RESUMEN GLOBAL", "📊 Superficie 3D", "🛡️ Cortante", 
    "🎨 Detallado Alzado", "🧱 Despiece BIM (2D) y BOM"
])

with t_res: st.dataframe(df_master, use_container_width=True)

with t_3d:
    fig = go.Figure()
    fig.add_trace(go.Surface(x=MX_GRID, y=MY_GRID, z=P_GRID, colorscale='Teal', opacity=0.4))
    if not edit_df.empty: fig.add_trace(go.Scatter3d(x=edit_df["Mux [Ton-m]"], y=edit_df["Muy [Ton-m]"], z=edit_df["Pu [Ton]"], mode='markers+text', text=edit_df["Combo"], marker=dict(size=6, color='red')))
    fig.update_layout(scene=dict(xaxis_title="Mux", yaxis_title="Muy", zaxis_title="Pu"), height=600)
    st.plotly_chart(fig, use_container_width=True)

with t_cort:
    c1, c2 = st.columns(2)
    c1.info(f"**Eje X:** VRx = {VR_x:.2f} Ton (Vcr={Vcr_x:.1f}, Vsr={Vsr_x:.1f})")
    c2.info(f"**Eje Y:** VRy = {VR_y:.2f} Ton (Vcr={Vcr_y:.1f}, Vsr={Vsr_y:.1f})")

with t_det:
    fig_p = go.Figure()
    w_draw = b if forma == "Rectangular/Cuadrada" else D
    fig_p.add_shape(type="rect", x0=-w_draw/2, y0=0, x1=w_draw/2, y1=H_cm, fillcolor="#EAECEE", line=dict(color="black", width=2))
    for py in posiciones_y: fig_p.add_shape(type="line", x0=-w_draw/2+rec, y0=py, x1=w_draw/2-rec, y1=py, line=dict(color="#C0392B", width=2))
    fig_p.update_layout(title=f"Alzado de estribos (s = {separacion_s} cm)", yaxis_title="Altura [cm]", height=500)
    st.plotly_chart(fig_p, use_container_width=True)

# --- PESTAÑA 5: PASO 3 (BIM 2D) + CUBICACIÓN ---
with t_gen:
    st.subheader("Despiece Longitudinal Interactivo y Sección Transversal")
    
    min_bars = 4 if forma == "Rectangular/Cuadrada" else 6
    def_6 = max(min_bars, int(np.ceil(As / CATALOGO_VARILLAS["#6 (3/4\")"]["area"])))
    if def_6 % 2 != 0: def_6 += 1 

    c_izq, c_der = st.columns([3, 2])
    
    with c_izq:
        st.write("**Selecciona el número de varillas:**")
        cv1, cv2, cv3, cv4, cv5 = st.columns(5)
        nv4 = cv1.number_input("#4 (1/2\")", 0, 40, 0)
        nv5 = cv2.number_input("#5 (5/8\")", 0, 40, 0)
        nv6 = cv3.number_input("#6 (3/4\")", 0, 40, def_6)
        nv8 = cv4.number_input("#8 (1\")", 0, 40, 0)
        nv10 = cv5.number_input("#10", 0, 40, 0)
        
        as_colocado = sum([nv4*1.27, nv5*1.99, nv6*2.85, nv8*5.07, nv10*7.92])
        if as_colocado < As: st.error(f"❌ Faltan {As - as_colocado:.2f} cm² de acero.")
        else: st.success(f"✅ Armado cubierto: {as_colocado:.2f} cm² colocados.")

        # Math BOM
        peso_long = (nv4*0.994 + nv5*1.56 + nv6*2.235 + nv8*3.975 + nv10*6.225) * (Lu + 0.85)
        L_est_m = (2*(b - 2*rec + h - 2*rec) + 20) / 100 if forma=="Rectangular/Cuadrada" else (np.pi*(D-2*rec)+20)/100
        peso_trans = num_estribos_total * L_est_m * CATALOGO_VARILLAS[tipo_estribo]["peso"]
        gran_total_kg = (peso_long + peso_trans) * 1.05 # 5% desperdicio
        ratio_kg_m3 = gran_total_kg / ((Ag * Lu)/10000)

    # PASO 3: DIBUJO SECCIÓN TRANSVERSAL 2D (BIM)
    with c_der:
        total_varillas_num = nv4 + nv5 + nv6 + nv8 + nv10
        fig_2d = go.Figure()
        
        # 1. Dibujar Concreto
        if forma == "Rectangular/Cuadrada":
            fig_2d.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, fillcolor="#E5E7E9", line=dict(color="#2C3E50", width=3))
            fig_2d.add_shape(type="rect", x0=rec, y0=rec, x1=b-rec, y1=h-rec, line=dict(color="#E74C3C", width=2, dash="dash"))
        else:
            fig_2d.add_shape(type="circle", x0=0, y0=0, x1=D, y1=D, fillcolor="#E5E7E9", line=dict(color="#2C3E50", width=3))
            fig_2d.add_shape(type="circle", x0=rec, y0=rec, x1=D-rec, y1=D-rec, line=dict(color="#E74C3C", width=2, dash="dash"))

        # 2. Repartir coordenadas de varillas proporcionalmente en el perímetro del estribo
        vx_c, vy_c = [], []
        if total_varillas_num > 0:
            if forma == "Rectangular/Cuadrada":
                W_est, H_est = b - 2*rec, h - 2*rec
                P_est = 2*W_est + 2*H_est
                for i in range(total_varillas_num):
                    d = i * (P_est / total_varillas_num)
                    if d <= W_est: vx_c.append(rec + d); vy_c.append(rec)
                    elif d <= W_est + H_est: vx_c.append(rec + W_est); vy_c.append(rec + (d - W_est))
                    elif d <= 2*W_est + H_est: vx_c.append(rec + W_est - (d - (W_est + H_est))); vy_c.append(rec + H_est)
                    else: vx_c.append(rec); vy_c.append(rec + H_est - (d - (2*W_est + H_est)))
            else:
                R_est = (D/2) - rec
                for i in range(total_varillas_num):
                    ang = i * (2 * np.pi / total_varillas_num)
                    vx_c.append((D/2) + R_est * np.cos(ang))
                    vy_c.append((D/2) + R_est * np.sin(ang))

            fig_2d.add_trace(go.Scatter(x=vx_c, y=vy_c, mode="markers", marker=dict(size=12, color="#2C3E50", line=dict(color="white", width=2)), name="Varillas"))

        max_d = max(b, h) if forma=="Rectangular/Cuadrada" else D
        fig_2d.update_layout(title="Corte Transversal (Vista superior)", xaxis=dict(range=[-5, max_d+5], scaleanchor="y", scaleratio=1, visible=False), yaxis=dict(range=[-5, max_d+5], visible=False), height=350, showlegend=False)
        st.plotly_chart(fig_2d, use_container_width=True)

    st.write(f"**Materiales:** Acero Total: **{gran_total_kg:.1f} kg** | Ratio de armado: **{ratio_kg_m3:.1f} kg/m³ de concreto**")


# --- PASO 1: FUNCIÓN GENERADORA DE MEMORIA PDF ---
def generar_memoria_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    
    # Encabezado
    pdf.set_font('Arial', 'B', 15)
    pdf.cell(0, 10, 'MEMORIA DE CÁLCULO ESTRUCTURAL - COLUMNA', ln=True, align='C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, 'Reglamento: NTC Concreto 2023 (CDMX)', ln=True, align='C')
    pdf.line(15, 27, 195, 27)
    pdf.ln(10)
    
    # Bloque 1: Geometría
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, '1. ESPECIFICACIONES DE GEOMETRÍA Y MATERIALES', ln=True)
    pdf.set_font('Arial', '', 10)
    sec_str = f"{b} x {h} cm" if forma == "Rectangular/Cuadrada" else f"Diámetro = {D} cm"
    pdf.cell(0, 6, f"   - Sección: {forma} ({sec_str})", ln=True)
    pdf.cell(0, 6, f"   - Altura libre (Lu): {Lu} m     |     Recubrimiento: {rec} cm", ln=True)
    pdf.cell(0, 6, f"   - Concreto f'c: {fc} kg/cm2     |     Acero fy: {fy} kg/cm2", ln=True)
    pdf.ln(5)
    
    # Bloque 2: Tabla de Resultados
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, '2. VERIFICACIÓN DE ESTADOS LÍMITE (ETABS)', ln=True)
    
    pdf.set_font('Arial', 'B', 9)
    col_w = [30, 28, 28, 28, 45]
    headers = ["Combo", "D/C Flexion", "D/C Cortante", "Deriva Max", "Estatus"]
    for i, head in enumerate(headers): pdf.cell(col_w[i], 7, head, border=1, align='C')
    pdf.ln()
    
    pdf.set_font('Arial', '', 9)
    for idx, row in df_master.iterrows():
        estatus_texto = "APROBADO" if "APROBADO" in str(row["Estatus"]) else "RECHAZADO"
        pdf.cell(col_w[0], 6, str(row["Combo"]), border=1, align='C')
        pdf.cell(col_w[1], 6, str(row["D/C Flexión"]), border=1, align='C')
        pdf.cell(col_w[2], 6, str(row["D/C Cortante"]), border=1, align='C')
        pdf.cell(col_w[3], 6, str(row["Deriva Máx"]), border=1, align='C')
        pdf.cell(col_w[4], 6, estatus_texto, border=1, align='C')
        pdf.ln()
    pdf.ln(8)
    
    # Bloque 3: BOM
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, '3. CUANTIFICACIÓN DE ACERO (BOM)', ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"   - Armado colocado: {total_varillas_num} varillas en total (As = {as_colocado:.2f} cm2)", ln=True)
    pdf.cell(0, 6, f"   - Estribado: Calibre {tipo_estribo} @ {separacion_s} cm ({num_estribos_total} camas)", ln=True)
    pdf.cell(0, 6, f"   - PESO TOTAL DE ACERO: {gran_total_kg:.2f} kg (Incluye 5% de merma)", ln=True)
    pdf.cell(0, 6, f"   - Indice de Congestion: {ratio_kg_m3:.1f} kg/m3 de concreto", ln=True)

    return pdf.output(dest='S')

# --- BOTÓN DE DESCARGA PDF EN BARRA LATERAL ---
st.sidebar.divider()
if FPDF_DISPONIBLE:
    pdf_bytes = generar_memoria_pdf()
    st.sidebar.download_button(label="🖨️ Descargar Memoria PDF", data=pdf_bytes, file_name="Memoria_Columna_NTC.pdf", mime="application/pdf", type="primary")
else:
    st.sidebar.caption("⚠️ Instala `pip install fpdf2` para activar el PDF.")
