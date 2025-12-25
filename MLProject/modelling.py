import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

# --- KONFIGURASI DAGSHUB (HARDCODED) ---
DAGSHUB_USERNAME = "wisinau"
DAGSHUB_REPO_NAME = "Heart-Failure-Tracking"
DAGSHUB_TOKEN = "76fe4c9242852a6125f39c059bc538163d4a236e"

# Setup Autentikasi
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

# --- LOAD DATA (Path diperbaiki agar terbaca Robot) ---
# Nama folder harus sesuai dengan yang ada di repo
DATA_PATH = "heart_failure_preprocessing"

def load_data():
    print(f"[INFO] Mencari data di folder: {os.getcwd()}/{DATA_PATH}")
    
    # Cek apakah folder ada
    if not os.path.exists(DATA_PATH):
        # Fallback: Coba cari di current directory (siapa tahu strukturnya beda)
        print("[WARNING] Folder tidak ditemukan, mencoba mencari di root...")
        DATA_PATH_ALT = "."
    else:
        DATA_PATH_ALT = DATA_PATH

    try:
        X_train = pd.read_csv(os.path.join(DATA_PATH_ALT, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(DATA_PATH_ALT, "y_train.csv")).values.ravel()
        X_test = pd.read_csv(os.path.join(DATA_PATH_ALT, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(DATA_PATH_ALT, "y_test.csv")).values.ravel()
        print("[SUCCESS] Data berhasil dimuat!")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"[FATAL ERROR] File CSV tidak ditemukan di {DATA_PATH_ALT}")
        print("Pastikan folder 'heart_failure_preprocessing' beserta isinya sudah ter-upload!")
        raise

def main():
    print(f"[INFO] Tracking ke DagsHub: {DAGSHUB_REPO_NAME}")
    mlflow.set_experiment("CI_CD_Automation_Aditya")
    
    print("[INFO] Loading Data...")
    X_train, y_train, X_test, y_test = load_data()
    
    with mlflow.start_run(run_name="GitHub_Action_Run_Fix"):
        print("[INFO] Training Model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"Metrics: Acc={acc}, F1={f1}")

        # Logging ke DagsHub
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        
        print("[SUCCESS] Training Selesai & Terkirim ke DagsHub!")

if __name__ == "__main__":
    main()
