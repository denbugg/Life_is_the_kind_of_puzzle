# Novel scene-analog layout: preregistration and result

Дата preregistration: 2026-08-29, до первого calibration run.

## Почему это новый механизм

Проверенная ранее source retrieval искала точную исходную фотографию, а
tile-only absolute heads пытались предсказать координату каждого фрагмента без
условия на конкретную сцену. Здесь используется промежуточная, deployable
гипотеза: **несколько похожих, но не совпадающих clean train-сцен служат
пространственными аналогами текущего unordered bag**.

1. Dirty board кодируется распределением full-tile color/texture/4×4-shape
   признаков; сигнатура инвариантна к permutation.
2. На paired train-only данных фиксированный ridge bridge переводит dirty
   сигнатуру в clean-reference domain.
3. Из source-disjoint train library выбираются ближайшие clean analogs.
4. Каждый analog задаёт стоимость `query tile → spatial slot`; стоимости
   нескольких analogs усредняются и единожды проектируются Hungarian в строгую
   one-to-one раскладку.
5. Рендер использует только исходные dirty tiles, затем frozen colored NLM
   `h=9`.

Это не seam-only scorer, не set-to-grid CE, не старый content verifier и не
solver-only преобразование старой матрицы. Scene choice зависит от всего bag,
full-tile evidence нелокально задаёт 576×576 position cost, а биекция является
частью механизма.

## Frozen cheap gate

- manifest: `data/interim/validation_manifest.json`;
- protocol digest:
  `2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`;
- train library: 512 records, SHA-ranked namespace
  `novel-analog-layout-library-v1`, seed `20260829`;
- evaluation: 24 calibration records, namespace
  `novel-analog-layout-calibration-v1`, seed `20260829`;
- fixed bridge: ridge `alpha=10`, без sweep;
- primary arm: top-4 distance-weighted analog consensus + colored NLM `h=9`;
- controls: unchanged shuffled input + NLM, generic mean template + NLM,
  top-1 analog + NLM and top-8 analog consensus + NLM;
- target pixels are used only by the official SSIM evaluator after every
  inference decision.

Primary gate, объявленный заранее:

1. mean paired SSIM gain primary vs shuffled-input+NLM `>= +0.015`;
2. primary mean SSIM `>= 0.18`;
3. paired 95% bootstrap interval lower bound `> 0`;
4. primary wins on at least 16/24 boards.

Holdout запрещено открывать, если хотя бы одно условие не выполнено. Controls
не могут заменить primary arm post hoc. Authoritative output должен сохранить
filenames, selection digests, hashes, per-board metrics и gate booleans.

## Result

Authoritative output:
`outputs/novel-analog-layout/calibration24.json`.

- library selection digest:
  `e30bd85cba669f24c71b8fecbe0d4032cea243bd8e1e1609b7fd5c7e7bdaaaa0`;
- calibration selection digest:
  `ed8c392cbed13907ee9a4b6f36548a6a59f3dc9507a660d1d9350af3654d3c63`;
- library/bridge runtime `12.22 s`, total runtime `33.16 s` on CPU;
- all 512 train pairs and 24 calibration pairs passed manifest hash checks.

| Arm | Mean SSIM | Gain vs input+NLM | 95% paired bootstrap CI | Wins |
|---|---:|---:|---:|---:|
| shuffled input, raw | 0.091468 | −0.053005 | [−0.061808, −0.044858] | 0/24 |
| shuffled input + NLM | 0.144473 | baseline | — | — |
| generic population atlas, raw | 0.104581 | −0.039892 vs input+NLM | [−0.048248, −0.031833] | 0/24 |
| generic population atlas + NLM | **0.173026** | **+0.028554** | **[+0.023295, +0.033692]** | **24/24** |
| top-1 analog + NLM | 0.167328 | +0.022856 | [+0.018492, +0.027284] | 24/24 |
| **top-4 analog consensus + NLM (primary)** | **0.164152** | **+0.019679** | **[+0.014662, +0.024527]** | **23/24** |
| top-8 analog consensus + NLM | 0.166974 | +0.022501 | [+0.017316, +0.027843] | 23/24 |

Primary прошёл три из четырёх условий: gain, CI и wins. Он провалил заранее
зафиксированный absolute gate `mean >= 0.18` (`0.164152`), поэтому verdict —
**reject-as-tested**, а holdout не открыт.

## Механистический вывод

Глобальный non-seam signal реален: даже до NLM population atlas улучшает raw
layout над shuffled input на `+0.013113`, CI `[+0.010481,+0.015642]`, 23/24
wins. После NLM gain становится `+0.028554`, 24/24. Но retrieval конкретных
похожих сцен не добавляет информацию: generic atlas превосходит primary top-4
на `+0.008875`, CI `[+0.003687,+0.014350]`, 19/24.

Следовательно, гипотеза «nearest scene analogs дают полезную scene-specific
геометрию» закрыта в этой реализации. Population atlas сохраняется как новый
дешёвый **absolute unary**: его разумный следующий тест — заранее фиксированная
слабая примесь в score-matched сильный local solver на новой calibration panel,
а не самостоятельный layout и не sweep числа analogs на тех же 24 boards.
