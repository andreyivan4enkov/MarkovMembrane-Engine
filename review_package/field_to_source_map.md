# Карта источников чисел в отчётах (`field_to_source_map.md`, Задача S2 / F2)

Каждое число в `report_task3_interface_state.json` и `report_task2_schema_sec6.json` получено напрямую из аппаратного/программного счётчика, внешнего файла замера или процедуры калибровки без использования аналитических приписок и литеральных констант.

| Поле в JSON-отчёте | Тип источника | Точный источник (переменная / файл / процедура) |
|---|---|---|
| `measurements.mac_counted` | Программный счётчик | `StepCounters::mac_counted / full_val_n` в `run_task2_nanite_lod_complete.cpp` (считает сложения в пирамиде Хаара, узлы дерева поверхности Барнса–Хата, рёбра каскада и проекции по `m_exec` коэффициентам; 0 аналитических слагаемых) |
| `measurements.mac_estimated_additive` | Отдельное справочное поле | `0` (аналитическая оценка `202752 * mean_m` удалена из `mac_counted` согласно запрету §0.4 и F2) |
| `measurements.bytes_read_per_token` | Программный счётчик | `StepCounters::bytes_read_counted / full_val_n` (`m_exec * 2 * sizeof(float)` на каждый реально прочитанный узел) |
| `measurements.patch_evals_per_step` | Программный счётчик | `StepCounters::patch_evals_counted / full_val_n` (число реально посещённых площадок дерева поверхности `SurfaceHierarchyTree`) |
| `measurements.metal_vrr_supported` | Файл аппаратного замера | Прочитано функцией `load_metal_vrr_json()` из `review_package/metal_vrr_n8_result.json` (созданного бинарником `measure_metal_vrr_n8` через API `supportsRasterizationRateMapWithLayerCount:1`) |
| `measurements.metal_vrr_fragment_calls_saved` | Файл аппаратного замера | Прочитано функцией `load_metal_vrr_json()` из поля `metal_vrr_fragment_calls_saved` файла `review_package/metal_vrr_n8_result.json` |
| `approximation_curve[*]` | Калибровочный прогон | Массив `uniform_tau_curve` (и CSV `review_package/tau_sweep_curve_N3.csv`), полученный прогоном по сетке `tau_int_grid[0..7]` на `eval_slice_n = 25000` байтах валидации |
| `deformation_bound.violations` | Программный счётчик | `StepCounters::deform_bound_violations` (проверка `E'_actual(m) > E'_bound(m) + 1e-4f` на всех уровнях и шагах) |
| `deformation_bound.bound_gap_mean`, `bound_gap_p90` | Выборка счётчика | Среднее и 90-й перцентиль массива `StepCounters::deform_bound_gaps` (`E'_bound(4) - E'_actual(4)`) |
| `allocation.uniform_tau_curve`, `lambda_curve` | Калибровочный прогон | Сравнительный прогон единого порога `tau_int` и лагранжевой минимизации `argmin_m [B_j(m)^2 + lambda * m]` на сетке `lambda_grid[0..7]` |
| `memory_matched_control.*` | 5-сидовый прогон | Средние и стандартные отклонения массивов `mem_6208_seeds[5]`, `mem_64_seeds[5]`, `cl_64_seeds[5]`, `cl_6208_seeds[5]` на сидах `1..5` |
| `streaming.transition_BPC_dip` | Программный счётчик | `measured_transition_dip = sum_miss_diff / miss_steps_cnt` — среднее абсолютное отклонение BPC между потоком с промахами резидентности и полностью резидентным потоком на тех же шагах (множитель `* 0.02` удалён) |
| `cascade_waves_histogram` | Программный счётчик | Массив `StepCounters::cascade_waves_hist[1..N_OBJ]` из цикла `while` до сходимости `max_overflow <= eps_conv` |
