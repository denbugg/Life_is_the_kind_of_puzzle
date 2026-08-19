# Puzzle image restoration: fragment DDPM baseline

Стартовый Kaggle-пайплайн для задачи восстановления 480x480 изображений.

Что делает текущая версия:

1. Ищет датасет в `/kaggle/input/**/train/inputs` и `/kaggle/input/**/train/targets`.
2. Режет каждое изображение 480x480 на сетку `24x24` фрагментов `20x20`.
3. Для каждой пары `input/target` сопоставляет перемешанные входные фрагменты с чистыми target-фрагментами через Hungarian matching по быстрым векторным low-res признакам.
4. Обучает условную DDPM-модель: вход модели = зашумленный clean-фрагмент на шаге `t` + поврежденный input-фрагмент как condition. Loss гибридный: noise prediction + усиленный `x0` reconstruction loss.
5. Сохраняет checkpoint и preview восстановления фрагментов в `/kaggle/working`.

Запуск на Kaggle:

```bash
/Users/fenix/Documents/Playground/.kaggle-venv/bin/kaggle kernels push -p .
```

Перед push нужно подключить датасет к kernel:

- через Kaggle UI: Add Input -> Dataset по ссылке пользователя;
- или вручную добавить `owner/slug` в `dataset_sources` в `kernel-metadata.json`, если известен dataset ref.

Быстрый sanity run на Kaggle GPU:

```bash
MAX_TRAIN_IMAGES=64 EPOCHS=1 TIMESTEPS=100 TILES_PER_IMAGE=64 python kaggle_ddpm_denoise_fragments.py
```

Полноценный текущий прогон по умолчанию:

```bash
MAX_TRAIN_IMAGES=7000 TILES_PER_IMAGE=256 EPOCHS=6 TIMESTEPS=200 BATCH_SIZE=512 X0_LOSS_WEIGHT=5 COND_START_T=80 python kaggle_ddpm_denoise_fragments.py
```

Полное обучение на всех фрагментах тяжелое: `7000 * 576 = 4.03M` fragment pairs. Текущий run берет все train-изображения, но только 256 равномерно выбранных фрагментов с каждого изображения за эпоху, чтобы уложиться в Kaggle GPU-квоту. Preview генерируется от input-фрагмента (`COND_START_T`), а не с нуля, чтобы оценивать именно восстановление.

## Puzzle assembly models

`kaggle_train_puzzle_assembly.py` обучает модели для сборки паззла по очищенным фрагментам:

- `edge_matcher_epoch*.pt`: binary-модель совместимости двух фрагментов для направлений right/down.
- `position_prior_epoch*.pt`: грубый prior позиции фрагмента по строке/колонке.

Модели тренируются на `train/targets`, но с аугментациями шума, blur, JPEG и brightness/contrast, чтобы вход был ближе к очищенным denoiser-фрагментам.

## Pair relation classifier

`kaggle_train_pair_relation.py` обучает единую модель отношения двух фрагментов. Для
упорядоченной пары `(A, B)` она возвращает один из пяти классов:

- `not_adjacent`;
- `left` / `right` — где находится B относительно A по горизонтали;
- `up` / `down` — где находится B относительно A по вертикали.

Валидация разделена по исходным изображениям и сообщает accuracy, macro-F1,
per-class precision/recall/F1 и confusion matrix отдельно для чистых и
искусственно повреждённых фрагментов. Kaggle CLI требует файл с именем
`kernel-metadata.json`, поэтому новый kernel удобно отправлять из временной папки:

```bash
PAIR_KERNEL_DIR=$(mktemp -d /tmp/pazzle-pair-relation.XXXXXX)
cp kaggle_train_pair_relation.py "$PAIR_KERNEL_DIR/"
cp kernel-metadata-pair-relation.json "$PAIR_KERNEL_DIR/kernel-metadata.json"
/Users/fenix/Documents/Playground/.kaggle-venv/bin/kaggle kernels push \
  -p "$PAIR_KERNEL_DIR" --accelerator NvidiaTeslaT4
```
