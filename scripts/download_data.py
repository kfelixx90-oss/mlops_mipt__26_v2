"""
Скрипт для загрузки данных.
.
"""

import urllib.request
import zipfile
from pathlib import Path


def download_data():
    """Загружает данные для проекта."""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Pагрузка маленького датасета
    url = "https://www.kaggle.com/datasets/adhoppin/hand-gestures-dataset/download?datasetVersionNumber=1"
    output_path = data_dir / "dataset.zip"

    print(f"Загрузка данных из {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Данные сохранены в {output_path}")

        # Распаковка
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Распаковка завершена")

        # Удаление zip-архива
        output_path.unlink()

    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        print("Убедитесь, что у вас есть доступ к интернету")


if __name__ == "__main__":
    download_data()
