# E24: возобновление эксперимента и генерация submission ZIP

Актуально для generation 3. Все тяжёлые и временные файлы должны находиться
только на диске `E:`. Фрагменты всегда upright: вращения и отражения запрещены.

## Фактический итог generation 3 — STOP

13 августа 2026 года OOF structural evaluation завершился корректно, но научный
gate **не прошёл**. Отчёт:

`E:\pazzle_work\posegraph_e24_selector\contextual_relation_selector_oof_v1.json`

SHA256:

`29d0c455bd0b07ed7b3b43d7662eed814a8885f4de02bb43019ebb8ae812673b`

Решение: `kill_crs_v1`. Основные значения:

- mean proposed precision `0.1490766934` при gate `0.70`;
- worst proposed precision `0.1041666667` при gate `0.60`;
- mean true-relation recall `0.1636625409` при gate `0.65`;
- worst recall `0.1148105626` при gate `0.50`;
- mean exact-connected coverage `0.0049913194` при gate `0.50`;
- worst coverage `0.0017361111` при gate `0.35`;
- mean cycle-rank ratio `0.0068789377` при gate `0.05`.

Все integrity/orientation/fold/geometry checks прошли, поэтому это научный FAIL
селектора, а не повреждённый запуск. По замороженному протоколу **нельзя**
запускать E24 staged SSIM/NLM, final-all8, E25 или production ZIP. Команды ниже
сохраняются как инструкция воспроизведения и как готовый production интерфейс
для будущего нового эксперимента, который действительно получит все PASS.

## Замороженная идентичность E24

- репозиторий: `C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed`
- рабочий каталог: `E:\pazzle_work\posegraph_e24_selector`
- ledger: `E:\pazzle_work\posegraph_e24_selector\preflight\e24_crs_v1_preflight.json`
- ledger SHA256: `e859edfaff913329429115ad171571b8f5a40a3698a1c4a847f0abef1a5a4bf5`
- run contract: `6fe34603e714776dff53a763eab63abadedd85be0333b4d0861f0ad37f4fcbcc`
- canary scene 17: PASS

Нельзя повторно создавать preflight, менять E24 core/evaluator/runner, параметры
LightGBM, folds, признаки, decoder или gates. После сбоя повторяется ровно та же
атомарная команда.

## Общая настройка PowerShell

```powershell
Set-Location 'C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed'

$E24 = 'E:\pazzle_work\posegraph_e24_selector'
$Ledger = "$E24\preflight\e24_crs_v1_preflight.json"
$LedgerSha = 'e859edfaff913329429115ad171571b8f5a40a3698a1c4a847f0abef1a5a4bf5'

$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Ledger).Hash.ToLower()
if ($actual -ne $LedgerSha) { throw "E24 ledger SHA drift: $actual" }

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = "$E24\pycache"
$env:TEMP = "$E24\tmp"
$env:TMP = "$E24\tmp"
$env:TMPDIR = "$E24\tmp"
$env:JOBLIB_TEMP_FOLDER = "$E24\tmp"
$env:LIGHTGBM_TMPDIR = "$E24\tmp"
```

## 1. Завершить structural OOF

На момент написания готовы inputs `10..17` и features
`10,11,12,13,14,15,17`. Следующая недостающая транзакция — feature scene 16:

```powershell
python -B .\src\run_e24_context_relation_selector.py feature-worker `
  --image 16 `
  --ledger $Ledger `
  --ledger-sha256 $LedgerSha
```

После появления `image_0016_receipt.json` безопаснее запускать folds отдельными
короткими командами, чтобы лимит одной shell-сессии не убил orchestrator:

```powershell
foreach ($fold in 0..3) {
  python -B .\src\run_e24_context_relation_selector.py prepare-fold-labels `
    --fold $fold --ledger $Ledger --ledger-sha256 $LedgerSha
  if ($LASTEXITCODE -ne 0) { throw "prepare-fold-labels $fold failed" }

  python -B .\src\run_e24_context_relation_selector.py train-fold `
    --fold $fold --ledger $Ledger --ledger-sha256 $LedgerSha
  if ($LASTEXITCODE -ne 0) { throw "train-fold $fold failed" }

  python -B .\src\run_e24_context_relation_selector.py predict-fold `
    --fold $fold --ledger $Ledger --ledger-sha256 $LedgerSha
  if ($LASTEXITCODE -ne 0) { throw "predict-fold $fold failed" }
}

python -B .\src\run_e24_context_relation_selector.py structural-eval `
  --ledger $Ledger --ledger-sha256 $LedgerSha
```

`structural-eval` создаёт отчёт, но не финальный cumulative resource receipt.
После четырёх create-once fold commits можно один раз запустить restart-safe
orchestrator: он перепроверит существующие артефакты, повторит их детерминированно
и опубликует receipt.

```powershell
python -B .\src\run_e24_context_relation_selector.py orchestrate `
  --ledger $Ledger --ledger-sha256 $LedgerSha
```

Ожидаемые файлы:

- `E:\pazzle_work\posegraph_e24_selector\folds_v1\fold_0..3\commit.json`
- `E:\pazzle_work\posegraph_e24_selector\contextual_relation_selector_oof_v1.json`
- `E:\pazzle_work\posegraph_e24_selector\oof_orchestration_receipt.json`

## 2. Проверить structural PASS

```powershell
$s = Get-Content -Raw "$E24\contextual_relation_selector_oof_v1.json" | ConvertFrom-Json
$r = Get-Content -Raw "$E24\oof_orchestration_receipt.json" | ConvertFrom-Json
$s.stage
$s.decision.passed
$s.decision.checks | Format-List
$s.summary | Format-List
$r.status
$r.checks | Format-List
$r.resource | Format-List
```

Продолжать можно только если одновременно:

- `stage == go_staged_end_to_end`;
- `decision.passed == true` и все structural checks истинны;
- receipt `status == pass` и все его checks истинны.

Hard gates: proposed precision mean/worst `0.70/0.60`, recall `0.65/0.50`,
exact-connected coverage `0.50/0.35`, mean cycle-rank ratio `0.05`, OOF CPU не
более 8 часов, RAM не более 16 GiB, E24 artifacts не более 8 GiB. При FAIL нельзя
ослаблять пороги или запускать staged/E25/submission.

## 3. Обязательная цепочка перед production

Даже structural PASS сам по себе не разрешает ZIP. Замороженный маршрут:

1. structural PASS + authenticated orchestration receipt;
2. staged board/SSIM/NLM PASS на сценах 10..17;
3. единственный final-all8 fit (seed 1234, те же 227 признаков, 256 trees);
4. отдельный source-group-disjoint E25 PASS на 48 запечатанных сценах;
5. production-parity replay scene 17;
6. только затем чтение 700 test PNG и генерация ZIP.

Текущий `run_e24_staged_ssim_nlm.py` умеет только безопасную label-free часть и
намеренно блокирует real metrics, пока не заморожен узкий clean-target broker.
Команды `freeze` и `prepare-label-free` до реализации broker запускать нельзя:
seal append-only и зафиксирует неполный source contract.

Также пока отсутствуют final-all8 writer/manifest и E25 runner/report. Поэтому
`infer_e24.py` правильно завершается fail-closed даже с `--dry-run`. Это не
команда, которую нужно обходить или патчить вручную.

## 4. Команда ZIP после реализации и PASS всех authority writers

Сначала проверка authority и inventory:

```powershell
python -B .\src\infer_e24.py --device cuda --dry-run
```

Затем production отрезками по 12 часов:

```powershell
python -B .\src\infer_e24.py --device cuda --max-runtime-seconds 43200
```

Если процесс корректно завершился кодом `75`, продолжать:

```powershell
python -B .\src\infer_e24.py --device cuda --max-runtime-seconds 43200 --resume
```

Повторять `--resume` до `status: completed`.

Финальный путь:

`E:\pazzle_work\submissions\e24_crs_v1\submission_e24_crs_v1.zip`

ZIP сначала создаётся как `.pending`, проходит проверку 700 root members, CRC и
SHA каждого PNG и только затем публикуется атомарно. Исходный baseline
`E:\pazzle_work\submission_rank96_v1.zip` никогда не перезаписывается.

Текущий последовательный production оценивается примерно в 90–110 часов для
682 generic изображений плюс byte-copy 18 verified overrides. Все outputs и
resume manifests находятся под `E:\pazzle_work\submissions\e24_crs_v1`.
