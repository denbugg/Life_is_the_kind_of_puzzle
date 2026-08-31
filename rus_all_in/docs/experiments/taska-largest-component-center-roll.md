# TASKA largest-component centre roll

Дата фиксации: 2026-08-31. Статус: **opened diagnostic negative; не
переносить и не sweep-ить**.

## Проверенная гипотеза

В frozen raw TASKA solver самый большой translation-consistent component на
всех `32/32` opened cases оказался нормализован к левому верхнему углу. Его
средний размер — `78.66` tiles (диапазон `38..155`), а средний bounding box —
`13.44 x 13.94`. Это позволило проверить простейшую буквальную версию
эвристики «самый уверенный силуэт поставить в центр» без обучения и без
content labels:

1. заново построить raw-компоненты из уже frozen dirty-only TASKA matrices и
   harvested edges;
2. выбрать единственный самый большой component;
3. сделать один целочисленный cyclic roll всего строгого layout так, чтобы
   центр его bounding box оказался в центре `24x24` board.

Target использовался только после построения layouts для offline scoring.
Преобразование не вращает, не деформирует и не заменяет tiles; результат
остаётся перестановкой всех 576 исходных upright fragments.

## Opened32

| Arm | Satisfied pairs | Recall | Exact tiles |
|---|---:|---:|---:|
| Raw TASKA | 334.71875 | 0.303187274 | 4.46875 |
| Largest-component centre roll | 323.53125 | 0.293053668 | 0.90625 |
| Delta | -11.18750 | -0.010133605 | -3.56250 |

Pair wins/ties/losses: `0/1/31`; exact wins/ties/losses: `8/11/13`.

## Вывод

Простейшее отождествление largest component с центральным лицом/силуэтом
неверно. На реальных сценах большой согласованный component часто относится к
фону или структуре, доходящей до границы изображения; его top-left
нормализация несёт полезный implicit frame prior. Центрирование также создаёт
два новых cyclic cut и теряет правильные соседства.

Закрыта именно формула **largest component -> whole-layout centre roll**.
Content-aware выбор другого component остаётся отдельной гипотезой, но должен
сначала показать inference-visible face/foreground signal; нельзя повторять
этот roll с соседними округлениями, shift weights или выбором второго по
размеру component на той же панели.

