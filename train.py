import argparse

from ml_pipeline.trainer import ModelManager
from ml_pipeline.exporter import Exporter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-config-path", type=str, help="Path to train conf file", required=True
    )
    parser.add_argument(
        "--export-config-path", type=str, help="Path to export conf file", required=False
    )

    args = parser.parse_args()

    trainer = ModelManager(args.train_config_path)
    trainer.train()

    # exporter = Exporter(args.export_config_path)
    # exporter.test_on_data()
