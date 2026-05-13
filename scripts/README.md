# Скрипты данных

## `prepare_iam_data.py` — IAM Words

База IAM лежит на Kaggle; скрипт **не качает** архив по ссылке сам — только распаковывает **уже скачанный** `.zip` и раскладывает файлы так, как ожидает `IAMDataset` в коде (`iam_data/iam_words/words.txt` и `iam_data/iam_words/words/...`).

### Шаги

1. Вот тут https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database/data скачать
2. Из **корня репозитория** выполни распаковку (подставь путь к своему архиву):

   ```bash
   python scripts/prepare_iam_data.py extract --archive "C:\путь\к\скачанному.zip"
   ```

   По умолчанию данные попадут в каталог `iam_data` рядом с корнем репозитория. Другой каталог:

   ```bash
   python scripts/prepare_iam_data.py extract --archive "C:\путь\к.zip" --dest "D:\data\iam_data"
   ```

3. Проверить, что всё на месте (без распаковки, если данные уже лежат):

   ```bash
   python scripts/prepare_iam_data.py verify
   ```

   С другим `--dest` — так же, как в `extract`.

### Если `extract` пишет, что не нашёл `words.txt` + `words/`

Внутри твоего архива может быть другая структура. Распакуй zip вручную, найди пару «`words.txt` и соседняя папка `words/`» и вручную положи их в `iam_data/iam_words/`, затем запусти `verify`.

# Соре у меня просто гит уже не тянет такие файлы коммитить совсем(
