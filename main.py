import sys

def run_ui():
    import subprocess
    # subprocess.run(["streamlit", "run", "app/ui.py"])
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/ui.py"])


def run_ingestion():
    from ingestion.pipeline import run_pipeline
    from core.settings import EMBEDDING_MODEL
    run_pipeline(embedding_model=EMBEDDING_MODEL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ui", "ingestion"], required=True)

    args = parser.parse_args()

    if args.mode == "ui":
        run_ui()
    elif args.mode == "ingestion":
        run_ingestion()
