# Аудит функции шага `step_membrane_N1_to_N13` (Задача S1 / F1)

Проверяемый файл: `review_package/run_task2_nanite_lod_complete.cpp` (строки `475`–`868`, состояние ДО внесения изменений Задания 3).

---

## 1. Расстояние мембрана↔объект: по 64 секторам или от усреднённой точки?

`[ПРОВЕРЕНО В КОДЕ]` Расстояние до каждого объекта `j` считается **от одной усреднённой 3D-точки `active_pos[3]`**, в которую предварительно сворачиваются все 64 сектора поверхности. Посекторные расстояния между 64 площадками поверхности и 96 объектами не вычисляются.

**Цитата (`review_package/run_task2_nanite_lod_complete.cpp`, строки `523`–`590`):**
```cpp
   523|    float active_pos[3] = {0.0f, 0.0f, 0.0f};
   524|    float active_norm[3] = {0.0f, 0.0f, 0.0f};
   525|    float w_sum = 1e-6f;
...
   565|            float fw = (0.05f + std::fabs(eps[d])) * rad_flux[d];
   566|            active_pos[0] += p_surf[d][0] * fw;
   567|            active_pos[1] += p_surf[d][1] * fw;
   568|            active_pos[2] += p_surf[d][2] * fw;
   569|            active_norm[0] += nx * fw;
   570|            active_norm[1] += ny * fw;
   571|            active_norm[2] += nz * fw;
   572|            w_sum += fw;
   573|        }
   574|    }
   575|    active_pos[0] /= w_sum; active_pos[1] /= w_sum; active_pos[2] /= w_sum;
   576|    float an_len = std::sqrt(active_norm[0] * active_norm[0] + active_norm[1] * active_norm[1] + active_norm[2] * active_norm[2] + 1e-12f);
   577|    active_norm[0] /= an_len; active_norm[1] /= an_len; active_norm[2] /= an_len;
...
   582|    for (int j = 0; j < N_OBJ; ++j) {
   583|        float dx = eng.obj_pos[j][0] - active_pos[0];
   584|        float dy = eng.obj_pos[j][1] - active_pos[1];
   585|        float dz = eng.obj_pos[j][2] - active_pos[2];
   586|        float dist = std::sqrt(dx * dx + dy * dy + dz * dz + 1e-12f);
   587|        float cos_v = std::max(0.0f, (active_norm[0] * dx + active_norm[1] * dy + active_norm[2] * dz) / dist);
   588|        init_e[j] = holo_norm * cos_v * std::exp(-dist / eng.graph.radii[j]);
   589|        cur_wave[j] = init_e[j];
   590|    }
```

---

## 2. Какие величины состояния поверхности доходят до решения о детализации, а какие отбрасываются?

`[ПРОВЕРЕНО В КОДЕ]`:
1. **Что доходит до функции выбора уровня `compute_nanite_cut_N1_N2`:**
   - Вектор `eps_tree[0..63]` (быстрое преобразование Хаара от вектора невязки `eps[d] = w_ext[d] - w_int[d]`);
   - Четыре вектора `rgbl_tree[0..3][0..63]` (быстрое преобразование Хаара от 4 каналов `proc_rgbl[d][0..3]`), свёрнутые со скалярным 4-фильтром объекта `obj_rgbl_flt[j][0..3]`;
   - Скалярная энергия узла после каскада `node_e = absorbed[j] + cur_wave[j]`, одинаково умножающая все 64 коэффициента Хаара: `tree_sig[k] = (eps_tree[k] + rgbl_k) * (0.25f + node_e)`.
2. **Что отбрасывается (теряется) до решения о детализации:**
   - Посекторные 3D-координаты `p_surf[d][0..2]` и посекторные 3D-нормали `(nx, ny, nz)` всех 64 площадок (они сжаты в одну точку `active_pos[3]` и один вектор `active_norm[3]` с весами ошибки `fw`);
   - Посекторный поток площади `rad_flux[d] = lum_L * dA_poly` (он использован только для взвешивания центроида `active_pos` в строке `565`, но **вообще не входит** ни в `proc_rgbl`, ни в `tree_sig[k]`, из-за чего площадь полигона `dA_poly` и экспонента объёма `learned_alpha` оказывались оторванными от амплитуды `tree_sig`!);
   - Собственное внутреннее состояние секторов `w_int[d]` и `c_int[d]` вне невязки `eps[d]`.

**Цитата (`review_package/run_task2_nanite_lod_complete.cpp`, строки `559`–`565` и `641`–`655`):**
```cpp
   559|            proc_rgbl[d][0] = col_R * lum_L;
   560|            proc_rgbl[d][1] = col_G * lum_L;
   561|            proc_rgbl[d][2] = col_B * lum_L;
   562|            proc_rgbl[d][3] = (lum_L - 1.0f) * eps_u;
   563|            rad_flux[d] = lum_L * dA_poly;
   564|
   565|            float fw = (0.05f + std::fabs(eps[d])) * rad_flux[d];
...
   641|    for (int j = 0; j < N_OBJ; ++j) {
   642|        float node_e = absorbed[j] + cur_wave[j];
   643|
   644|        /* Собираем сигнал узла j в базисе дерева Хаара */
   645|        float tree_sig[D_MODEL];
   646|        const float *flt = eng.obj_rgbl_flt[j];
   647|        for (int k = 0; k < D_MODEL; ++k) {
   648|            float rgbl_k = flt[0] * rgbl_tree[0][k] + flt[1] * rgbl_tree[1][k] + flt[2] * rgbl_tree[2][k] + flt[3] * rgbl_tree[3][k];
   649|            /* Взвешиваем на перспективную энергию узла */
   650|            tree_sig[k] = (eps_tree[k] + rgbl_k) * (0.25f + node_e);
   651|            saved_tree_sig[j][k] = tree_sig[k];
   652|        }
   653|
   654|        /* N1 + N2: Вычисляем монотонную ошибку хвоста E(m) и двусторонний срез */
   655|        CutDecision dec = compute_nanite_cut_N1_N2(tree_sig, tau, two_sided_cut, active_margin);
```

---

## 3. Участвуют ли `obj_in` / `obj_out` / `psi_field` в решении о детализации, или подключаются после?

`[ПРОВЕРЕНО В КОДЕ]` Собственные векторы признака `obj_in[j][0..63]` и `obj_out[j][0..63]` в решении о детализации `compute_nanite_cut_N1_N2` **не участвуют вообще** — они подключаются только **после** того, как уровень среза `m_exec` уже выбран (в строках `705` и `716`). Поле памяти объектов `psi_field` в `DeformableHaarMembraneEngine` отсутствовало (хранился только 64-мерный вектор `w_int[64]`).

**Цитата (`review_package/run_task2_nanite_lod_complete.cpp`, строки `655`–`720`):**
```cpp
   655|        CutDecision dec = compute_nanite_cut_N1_N2(tree_sig, tau, two_sided_cut, active_margin);
...
   663|        int m_chosen = std::min(dec.m_keep, force_max_m);
...
   704|            for (int k = 0; k < m_exec; ++k) {
   705|                proj += tree_sig[k] * eng.obj_in[j][k] * hill_gate;
   706|            }
...
   715|            for (int k = 0; k < m_exec; ++k) {
   716|                tree_resp[k] = act * eng.obj_out[j][k] * hill_gate * multi_level_dup;
   717|            }
```

---

## 4. Глубина каскада: сходимость или константа?

`[ПРОВЕРЕНО В КОДЕ]` Глубина каскада зашита константой — **ровно 3 итерации** (`for (int w = 0; w < 3; ++w)`).

**Цитата (`review_package/run_task2_nanite_lod_complete.cpp`, строки `592`–`608`):**
```cpp
   592|    for (int w = 0; w < 3; ++w) {
   593|        float next_w[N_OBJ] = {};
   594|        for (int i = 0; i < N_OBJ; ++i) {
   595|            float rem = std::max(0.0f, eng.obj_z_imp[i] - absorbed[i]);
   596|            float tak = std::min(cur_wave[i], rem);
   597|            float over = std::max(0.0f, cur_wave[i] - tak);
   598|            absorbed[i] += tak;
   599|            if (over > 1e-4f) {
   600|                for (const auto &ed : eng.graph.adj[i]) {
   601|                    float align = std::max(0.0f, active_norm[0] * ed.nx + active_norm[1] * ed.ny + active_norm[2] * ed.nz);
   602|                    next_w[ed.dst] += over * ed.cond * align;
   603|                    cnt.mac_ops_counter += 2;
   604|                }
   605|            }
   606|        }
   607|        for (int j = 0; j < N_OBJ; ++j) cur_wave[j] = next_w[j];
   608|    }
```

---

## 5. `eps`: по координатам или скалярная норма?

`[ПРОВЕРЕНО В КОДЕ]`:
1. При расчёте начального возбуждения объектов `init_e[j]` вектор `eps[0..63]` **сжат в скалярную норму** `holo_norm = std::sqrt((float)err_before_sq + 1e-12f)` (строки `580`, `588`).
2. При формировании `eps_tree[0..63]` вектор `eps[0..63]` раскладывается по координатам в дерево Хаара (строка `613`), но из-за того, что `node_e` зависит только от скаляра `holo_norm` и единого центроида `active_pos`, геометрия расхождения в отдельных секторах поверхности не связывается с положением конкретных 3D-объектов.
