# Распознавание рукописных цифр

Я попробовала сделать красивенькую штуку с распознованием в онлайне: можно либо рисовать, либо вставлять фотографии. Там пока модель от Насти best_model.pth

К сожалению, пока это можно запустить только локально, но я попробую на следующей неделе это доработать)

## Запуск на своём компьютере

```bash
# Переключиться на ветку analytics
git checkout analytics

# Установить зависимости
pip install streamlit streamlit-drawable-canvas torch torchvision numpy pillow scikit-image

# Запустить приложение
streamlit run src/digit_recognizer.py

