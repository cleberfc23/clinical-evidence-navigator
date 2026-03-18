def run_ui():
    import subprocess
    subprocess.run(["streamlit", "run", "app/ui.py"])


def run_ingestion():
    from ingestion.pipeline import run_pipeline
    run_pipeline()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ui", "ingestion"], required=True)

    args = parser.parse_args()

    if args.mode == "ui":
        run_ui()
    elif args.mode == "ingestion":
        run_ingestion()
