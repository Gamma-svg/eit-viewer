# EIT Reconstruction Viewer · TFG Defense

## Estructura
```
web_eit/
├── index.html                  ← La web (abrir en el navegador)
├── manifest.json               ← Generado automáticamente
├── generar_imagenes_eit.py     ← Script Python para generar imágenes
└── imgs/                       ← Imágenes generadas (generadas por el script)
    └── <phantom>/<exp>/<pos>/<device>/<patron>/<algo>/<freq>.png
```

## Paso 1 — Generar las imágenes

Ejecuta desde la carpeta del proyecto:

```bash
python generar_imagenes_eit.py \
    --datos  /ruta/a/EIT_Pruebas_Hechas \
    --salida ./imgs \
    --metodos svd jac greit bp \
    --comp abs
```

Opciones:
- `--metodos`  : svd, jac, greit, bp (puedes elegir los que quieras)
- `--comp`     : abs, real, imag
- `--datos`    : ruta a la carpeta EIT_Pruebas_Hechas

El script también genera `manifest.json` automáticamente.

## Paso 2 — Subir a GitHub Pages

1. Crea una cuenta en https://github.com
2. New repository → nombre: `eit-viewer` → Public
3. Sube todos los archivos (drag & drop en la web de GitHub)
4. Settings → Pages → Source: main → Save
5. Tu URL será: `https://<tu-usuario>.github.io/eit-viewer`

## Paso 3 — Generar el QR

Ve a https://www.qr-code-generator.com/ y pega la URL.
Descarga el QR en PNG/SVG e insértalo en tu presentación.

## Probar en local (sin subir a GitHub)

```bash
# Python 3
python -m http.server 8080
# Abre: http://localhost:8080
```

## Notas para la defensa

- La web funciona en cualquier móvil/tablet sin instalar nada
- El slider de frecuencia permite mostrar cómo cambia la reconstrucción con la frecuencia
- Puedes comparar ScioSpec vs KIT vs mACQ en tiempo real cambiando el dispositivo
- Los 4 algoritmos (SVD, JAC, GREIT, BP) se ven simultáneamente para el mismo experimento
- Clic en cualquier reconstrucción → zoom en pantalla completa
