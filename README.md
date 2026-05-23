# Handwriting-Recognition

Две нейронки в одном репозитории:

- **MNIST** — распознавание цифр (`src/neural_network.py`, `MNISTModel`);
- **IAM lines** — распознавание целых рукописных строк, CRNN + CTC (`src/text.py`, `CRNNModel`).

## Структура проекта

```
.
├── data/processed/          # MNIST (.npy) — небольшой, лежит прямо в репо
├── iam/                     # IAM lines, локально (~1.5 ГБ, в .gitignore)
│   ├── gt/lines.txt         # транскрипции
│   ├── lines/<wr>/<wr-fm>/  # картинки строк (.png)
│   └── splits/              # официальные trainset.txt и др. (опц.)
├── scripts/
│   ├── prepare_iam_data.py  # распаковка архивов IAM в iam/
│   ├── check_setup.py       # проверка окружения и данных
│   └── README.md
├── src/
│   ├── dataset.py           # MNISTDataset (для цифр)
│   ├── neural_network.py    # MNISTModel + Trainer + main() для MNIST
│   └── text.py              # IAM lines: датасет, CRNN, обучение, beam search, main()
├── requirements.txt
└── README.md
```

## Установка

```bash
python -m venv .venv
. .venv/bin/activate          # Linux/macOS/WSL
# .venv\Scripts\activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

## Цифры (MNIST)

Данные уже лежат в `data/processed/*.npy`. Запуск обучения:

```bash
python src/neural_network.py
```

## Текст (IAM lines)

### 1. Скачать архивы IAM

С официального сайта [FKI IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) (после регистрации):

- `lines.tgz` (~640 МБ) — картинки строк;
- `ascii.tgz` (~3 МБ) — транскрипции (`lines.txt`).

Опционально, без регистрации — официальные сплиты:

- [`largeWriterIndependentTextLineRecognitionTask.zip`](https://fki.tic.heia-fr.ch/static/zip/largeWriterIndependentTextLineRecognitionTask.zip) (~25 КБ).

### 2. Распаковать в `iam/`

```bash
python scripts/prepare_iam_data.py extract \
  --lines  "путь/к/lines.tgz" \
  --ascii  "путь/к/ascii.tgz" \
  --splits "путь/к/largeWriterIndependentTextLineRecognitionTask.zip"
```

Без `--splits` сплиты делаются 80/10/10 по строкам (см. `scripts/README.md`).

### 3. Проверить, что всё на месте

```bash
python scripts/check_setup.py
```

Должно вывести `все проверки прошли`. Среди прочего напечатает реальный пример строки из батча.

### 4. Запустить обучение

Из корня репо:

```bash
python src/text.py
```

Параметры по умолчанию (`BATCH_SIZE=16`, `NUM_EPOCHS=30`, `LR=1e-3`, `IMG_HEIGHT=64`) лежат в функции `main()` в `src/text.py`. Лучшая модель сохраняется в `best_crnn.pth` в текущем каталоге (он в `.gitignore`).

### Опционально: KenLM beam search

Если рядом с проектом положить файл `lm/english.arpa` и установить `pyctcdecode + kenlm` — `text.py` подхватит KenLM для beam search. По умолчанию используется встроенная биграммная LM из тренировочных текстов (никаких внешних зависимостей).

## Заметки

- IAM-архивы и веса моделей в репо **не коммитятся** (см. `.gitignore`).
- В `lines.txt` IAM слова разделяются `|`, скрипт сам приводит это к пробелам — поэтому модель учится на нормальном тексте.
- Для запуска `text.py` с `num_workers > 0` на Windows используйте PowerShell + venv с собранным под Windows PyTorch (или WSL).
