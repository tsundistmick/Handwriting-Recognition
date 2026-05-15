# Скрипты данных

## `check_setup.py` — проверка после установки

Из **корня репозитория**, с активированным venv (`pip install -r requirements.txt`):

```bash
python scripts/check_setup.py
```

Проверяет по очереди:

1. зависимости (`torch`, `numpy`, `Pillow`);
2. MNIST в `data/processed/` и `MNISTModel`;
3. IAM на диске (`words.txt`, папка `words/` с png);
4. split train/val по писателям;
5. `IAMDataset` — чтение одной картинки.

Если в конце `все проверки прошли` — окружение и лоадеры готовы к обучению.

---

## `prepare_iam_data.py` — распаковка IAM Words

База IAM на Kaggle; скрипт **не качает** архив сам — только распаковывает **уже скачанный** `.zip` в `iam_data/iam_words/`.

1. Скачать: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database/data  
2. Распаковать:

   ```bash
   python scripts/prepare_iam_data.py extract --archive "C:\путь\к\скачанному.zip"
   ```

3. Проверить окружение целиком:

   ```bash
   python scripts/check_setup.py
   ```

Другой каталог для данных:

```bash
python scripts/prepare_iam_data.py extract --archive "C:\путь\к.zip" --dest "D:\data\iam_data"
```

Только проверка файлов IAM (без PyTorch):

```bash
python scripts/prepare_iam_data.py verify
```

### Если `extract` не нашёл `words.txt` + `words/`

Распакуй zip вручную, найди пару «`words.txt` и соседняя папка `words/`» и положи в `iam_data/iam_words/`, затем снова `python scripts/check_setup.py`.
