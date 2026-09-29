import os
import sys
import math
import re
import torch
import gc
import transformers
import pandas as pd
import traceback  
from tqdm import tqdm

def eliminar_tildes(cadena):
    import unicodedata
    texto_limpio = "".join(c for c in unicodedata.normalize('NFD', cadena) if unicodedata.category(c) != 'Mn')
    texto_limpio = texto_limpio.replace('ñ', 'n').replace('Ñ', 'N')
    return texto_limpio

# ==========================================================
# 1. CONFIGURACIÓN DE ENTORNO Y RUTAS (~/experiment)
# ==========================================================
HOME = os.path.expanduser("~")
RUTA_EXPERIMENTO = os.path.join(HOME, "experiment")

csv_entrada_submissions = os.path.join(RUTA_EXPERIMENTO, "standardized-submissions-file.csv")
csv_salida_local = os.path.join(RUTA_EXPERIMENTO, "impacto_ia_tesis.csv")

print(f"Directorio del experimento configurado en: {RUTA_EXPERIMENTO}")

# ==========================================================
# 2. CARGA DE DATOS Y CONSTRUCCIÓN DINÁMICA DE RUTAS
# ==========================================================
if not os.path.exists(csv_entrada_submissions):
    print(f"Error: No se encontró el archivo '{csv_entrada_submissions}' en la carpeta de experimentos.")
    sys.exit(1)

print("Cargando archivo estandarizado de entregas...")
sub_file = pd.read_csv(csv_entrada_submissions)

def construir_ruta_codigo_con_sufijo_plano(fila):
    try:

        ruta_origen = eliminar_tildes(str(fila['archivo_origen']).strip())
        raw_nombre = str(fila['_id']).strip()

        nombre_limpio = raw_nombre.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        if not nombre_limpio.endswith('.py'):
            nombre_limpio = f"{nombre_limpio}.py"

        plataforma = "edunext" if "edunext" in ruta_origen.lower() else "open_edx"

        nombre_csv = os.path.basename(ruta_origen)
        
        nombre_tarea_code = nombre_csv.replace("_metadata.csv", "_code")

        raiz_local_submissions = os.path.join(HOME, "experiment", "data", "submissions", plataforma)
        ruta_final = os.path.join(raiz_local_submissions, 'code', nombre_tarea_code, nombre_limpio)
        
        return ruta_final
    except Exception as e:
        return f"Error_Ruta: {str(e)}"

sub_file['ruta_real_py'] = sub_file.apply(construir_ruta_codigo_con_sufijo_plano, axis=1)
df_total = sub_file.dropna(subset=['ruta_real_py', 'archive']).copy()

# ==========================================================
# 3. GESTIÓN DE CHECKPOINTS (EVITA REPETIR TRABAJO)
# ==========================================================
if os.path.exists(csv_salida_local):
    try:
        df_progreso_previo = pd.read_csv(csv_salida_local)
        archivos_ya_procesados = set(df_progreso_previo['archivo'].dropna().unique())
        print(f"Checkpoint detectado. Auditado antes: {len(archivos_ya_procesados)}")
        df_pendientes = df_total[~df_total['archive'].isin(archivos_ya_procesados)].copy()
    except Exception as e:
        print(f"Error leyendo progreso previo, iniciando limpio: {str(e)}")
        archivos_ya_procesados = set()
        df_pendientes = df_total.copy()
else:
    print("No se detectó progreso previo. Empezar de nuevo")
    archivos_ya_procesados = set()
    df_pendientes = df_total.copy()

print(f"Volumen total cargado: {len(df_total)} | Pendientes por procesar: {len(df_pendientes)}")

if len(df_pendientes) == 0:
    print("Terminado con exito")
    sys.exit()

# ==========================================================
# 4. CARGA GLOBAL DE MODELOS EN GPU (CONFIGURADOS EN LOTE)
# ==========================================================
sys.path.append(os.path.join(RUTA_EXPERIMENTO, 'DetectCodeGPT'))
try:
    from predict import get_log_likelihood
except ImportError:
    print(f"Error: No se encontró la carpeta fork '{os.path.join(RUTA_EXPERIMENTO, 'DetectCodeGPT')}'")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[Hardware] Motores asignados a: {device.upper()}")

base_model_name = "Salesforce/codegen-350M-mono"
mask_model_name = "Salesforce/codet5p-770m"

# Configuración para permitir procesamiento en lote (Batching)
base_tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
base_tokenizer.pad_token = base_tokenizer.eos_token

base_model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_model_name)
mask_tokenizer.pad_token = mask_tokenizer.eos_token

mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    mask_model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# ==========================================================
# ⚡ FUNCIÓN DE PERTURBACIÓN MASIVA POR LOTES (BATCH GENERATION)
# ==========================================================
def perturb_code_BATCH(lista_textos, mask_model, mask_tokenizer, device, n_perturbations=5):
    prompts = [f"Perturb open-source python code maintaining logic: {t}" for t in lista_textos]
    
    # Token sobre todo el grupo de estudiantes a la vez con padding y truncado preventivo por velocidad
    inputs = mask_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=100).to(device)

    with torch.no_grad():
        outputs = mask_model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=n_perturbations,
            pad_token_id=mask_tokenizer.pad_token_id
        )

    resultado_por_estudiante = []
    total_outputs = len(outputs)
    
    for k in range(len(lista_textos)):
        inicio_idx = k * n_perturbations
        fin_idx = min(inicio_idx + n_perturbations, total_outputs)
        
        mutaciones_estudiante = []
        for idx_out in range(inicio_idx, fin_idx):
            p_text = mask_tokenizer.decode(outputs[idx_out], skip_special_tokens=True).strip()
            if p_text and len(p_text) > 5:
                p_text = re.sub(r" <extra_id_\d+> ", "", p_text)
                mutaciones_estudiante.append(p_text)
        resultado_por_estudiante.append(mutaciones_estudiante)
        
    return resultado_por_estudiante

# ==========================================================
# 5. PIPELINE ULTRA-OPTIMIZADO CON RASTREO DE ERRORES (V5)
# ==========================================================
lote_guardado = []
tamano_guardado = 100
BATCH_ESTUDIANTES = 8  # 8 estudiantes en paralelo en la Titan X

print(f"\nProcesamiento en Paralelo (Batch Size = {BATCH_ESTUDIANTES})...")

registros_pendientes = df_pendientes.to_dict('records')
pbar = tqdm(total=len(registros_pendientes), desc="Auditando entregas masivas")

i = 0
while i < len(registros_pendientes):
    bloque = registros_pendientes[i : i + BATCH_ESTUDIANTES]
    i += len(bloque) 
    
    datos_validos = []
    textos_codigos = []
    
    for fila in bloque:
        ruta_archivo = fila['ruta_real_py']
        if "Error_Ruta" in str(ruta_archivo) or not os.path.exists(ruta_archivo):
            pbar.update(1)
            continue
        try:
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                codigo = f.read().strip()
            if codigo and len(codigo) >= 10:
                textos_codigos.append(codigo)
                datos_validos.append(fila)
            else:
                pbar.update(1)
        except Exception as e_lectura:
            print(f"Error leyendo archivo físico {ruta_archivo}: {str(e_lectura)}")
            pbar.update(1)
            continue

    if not textos_codigos:
        continue

    # ==========================================================
    # ZONA OPERATIVA EN GPU
    # ==========================================================
    try:
        with torch.no_grad():
            # Calcular Log-Likelihood originales
            ll_originales = [get_log_likelihood(c, base_model, base_tokenizer, device) for c in textos_codigos]

            # mutaciones en batch
            todas_las_mutaciones = perturb_code_BATCH(textos_codigos, mask_model, mask_tokenizer, device, n_perturbations=5)

        # Procesar y promediar los resultados obtenidos
        for idx, fila in enumerate(datos_validos):
            ll_orig = ll_originales[idx]
            mutaciones_del_estudiante = todas_las_mutaciones[idx]

            if ll_orig is None or not mutaciones_del_estudiante:
                pbar.update(1)
                continue

            with torch.no_grad():
                ll_mutados = [get_log_likelihood(m, base_model, base_tokenizer, device) for m in mutaciones_del_estudiante]
                ll_validos = [v for v in ll_mutados if v is not None]

            if len(ll_validos) == 0:
                pbar.update(1)
                continue

            ll_perturbed_avg = sum(ll_validos) / len(ll_validos)
            discrepancy = ll_orig - ll_perturbed_avg
            prob_ia = (1 / (1 + math.exp(-10 * (discrepancy - 0.05)))) * 100

            actividad = os.path.basename(os.path.dirname(fila['ruta_real_py']))
            
            lote_guardado.append({
                "archivo": fila['archive'],
                "actividad": actividad,
                "ano": fila.get('ano', 2026),
                "mes": fila.get('mes', 7),
                "LL_Original": round(ll_orig, 4),
                "LL_Mutados_Avg": round(ll_perturbed_avg, 4),
                "discrepancia_cruda": round(discrepancy, 4),
                "probabilidad_IA": round(prob_ia, 2)
            })
            pbar.update(1)

        if len(lote_guardado) >= tamano_guardado:
            df_lote = pd.DataFrame(lote_guardado)
            if os.path.exists(csv_salida_local):
                df_lote.to_csv(csv_salida_local, mode='a', header=False, index=False)
            else:
                df_lote.to_csv(csv_salida_local, mode='w', header=True, index=False)
            lote_guardado = []

    except Exception as e_gpu:
        print("\nOcurrió un error procesando el lote en la GPU:")
        print(f"Tipo de error: {type(e_gpu).__name__}")
        print(f"Mensaje: {str(e_gpu)}")
        print("Rastreo completo del error (Traceback):")
        traceback.print_exc() 
        print("-" * 60)
        
        pbar.update(len(bloque))
        
    torch.cuda.empty_cache()
    gc.collect()

if len(lote_guardado) > 0:
    df_lote = pd.DataFrame(lote_guardado)
    if os.path.exists(csv_salida_local):
        df_lote.to_csv(csv_salida_local, mode='a', header=False, index=False)
    else:
        df_lote.to_csv(csv_salida_local, mode='w', header=True, index=False)

pbar.close()
print(f"\n Proceso totalmente terminado {csv_salida_local}.")