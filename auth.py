#!/usr/bin/env python3
"""
auth.py — Genera token.pickle para la YouTube Data API v3.

Ejecutar UNA vez desde la máquina con navegador:
    python auth.py

Luego copiar token.pickle al servidor donde corre el contenedor.
"""

import pickle
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
creds = flow.run_local_server(port=0)

with open("token.pickle", "wb") as f:
    pickle.dump(creds, f)

print("[SUCCESS] token.pickle generado. Cópialo al servidor antes de levantar el contenedor.")
