# Скрипты данных

## `check_setup.py` — проверка после установки

Из **корня репозитория**, с активированным venv (`pip install -r requirements.txt`):

```bash
python scripts/check_setup.py
```

Проверяет по очереди:

1. зависимости (`torch`, `numpy`, `torchvision`, `Pillow`);
2. MNIST в `data/processed/` и `MNISTModel`;
3. IAM на диске (`iam/lines/`, `iam/gt/lines.txt`, `iam/splits/`);
4. сплиты train/val/test через `load_split_ids` из `src/text.py`;
5. `IAMDataset` + `collate_fn` — собирается ли один батч.

Если в конце «все проверки прошли» — можно запускать обучение:

```bash
python src/text.py
```

---

## `prepare_iam_data.py` — распаковка IAM (line-level)

Базовый источник — официальный [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) (FKI, University of Bern). Скрипт **не качает** архивы сам — только распаковывает уже скачанные `.tgz` / `.zip` в `iam/`.

### Что нужно скачать (после регистрации на сайте FKI)

| Архив | Зачем | Где |
|-------|-------|------|
| `lines.tgz` (~640 MB) | картинки строк → `iam/lines/` | `data/lines.tgz` на сайте |
| `ascii.tgz` (~3 MB) | `lines.txt` с транскрипциями → `iam/gt/lines.txt` | `data/ascii.tgz` |
| `largeWriterIndependentTextLineRecognitionTask.zip` (~25 KB, опц.) | официальные сплиты → `iam/splits/` | [прямая ссылка](https://fki.tic.heia-fr.ch/static/zip/largeWriterIndependentTextLineRecognitionTask.zip), без регистрации |

`forms*.tgz`, `words.tgz`, `sentences.tgz`, `xml.tgz` — **не нужны** для обучения по строкам.

### Распаковка

```bash
python scripts/prepare_iam_data.py extract \
  --lines  "C:/путь/к/lines.tgz" \
  --ascii  "C:/путь/к/ascii.tgz" \
  --splits "C:/путь/к/largeWriterIndependentTextLineRecognitionTask.zip"
```

`--splits` опционален. Без него `text.py` сам уйдёт на fallback **80/10/10** по строкам (без writer-independent — обучение пройдёт, метрики на val/test могут быть оптимистичнее «по статьям»).

Поменять каталог назначения (по умолчанию `iam/` в корне репо):

```bash
python scripts/prepare_iam_data.py extract --dest "D:/data/iam" --lines ... --ascii ...
```

### Только проверка (без распаковки)

```bash
python scripts/prepare_iam_data.py verify
```

Покажет, сколько строк со статусом `ok` в `lines.txt`, сколько `.png` в `lines/` и какие сплит-файлы найдены.

### Если что-то пошло не так

- **«в архиве не найдено картинок строк»** — проверьте, что это именно `lines.tgz` (внутри должна быть структура `<writer>/<writer-form>/<line-id>.png`, например `a01/a01-000u/a01-000u-00.png`).
- **«не найден lines.txt»** — нужен `ascii.tgz`, не `xml.tgz` и не `words.txt`.
- **сплиты пустые в `verify`** — внутри splits-zip ищутся файлы `trainset.txt`, `validationset1.txt`, `validationset2.txt`, `testset.txt`.
