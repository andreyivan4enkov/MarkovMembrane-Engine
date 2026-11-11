# Аудит функции шага `step_membrane_N1_to_N13` (Задача F1)

Проверяемый файл: `review_package/run_task2_nanite_lod_complete.cpp` (строки `475`–`868`, состояние ДО внесения изменений Задания 3).

---

## 1. Как считается расстояние мембрана↔объект: от каждого из 64 секторов или от одной усреднённой точки?

**Факт:** Расстояние считается **от одной усреднённой 3D-точки (`active_pos[3]`)**, в которую предварительно сворачиваются все 64 сектора поверхности с весами ошибки `fw = (0.05f + std::fabs(eps[d])) * rad_flux[d]`. Отдельные расстояния от 64 секторов или групп секторов поверхности до объекта не считаются.

**Цитата строк `523`–`590` (`review_package/run_task2_nanite_lod_complete.cpp`):**
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
   578|
   579|    /* Перспектива и каскад по списку соседей N7 (без N^2 обхода!) */
   580|    float holo_norm = std::sqrt((float)err_before_sq + 1e-12f);
   581|    float init_e[N_OBJ], absorbed[N_OBJ] = {}, cur_wave[N_OBJ];
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

## 2. Глубина каскада: до сходимости или фиксированная?

**Факт:** Глубина каскада **фиксированная — ровно 3 волны** (`for (int w = 0; w < 3; ++w)`), как в основной функции шага `step_membrane_N1_to_N13`, так и во вспомогательной функции `verify_neighbor_list_N7`. Критерия остановки по сходимости остатка переноса в этом цикле нет.

**Цитата строк `592`–`608` (`review_package/run_task2_nanite_lod_complete.cpp`):**
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

И аналогично в `verify_neighbor_list_N7` (строка `322`):
```cpp
   322|        for (int w = 0; w < 3; ++w) {
```

---

## 3. Ошибка `eps`: по координатам или свёрнута в скаляр norm?

**Факт:** Ошибка `eps` используется двояко:
1. **При возбуждении начальной энергии каскада (`init_e[j]`)** массив `eps[0..63]` **сворачивается в один скаляр `holo_norm = sqrt(err_before_sq)`** (строки `500`, `580`, `588`), из-за чего пространственное распределение ошибки по 64 секторам теряется при входе в каскад.
2. **При разложении сигнала в дерево Хаара (`eps_tree`)** используется покоординатный массив `eps[0..63]`, но затем в узле `j` все 64 коэффициента `eps_tree[k] + rgbl_k` умножаются на один и тот же скалярный множитель `(0.25f + node_e)` (строка `650`).

**Цитата строк `497`–`501`, `580`–`589` и `611`–`652` (`review_package/run_task2_nanite_lod_complete.cpp`):**
```cpp
   497|    for (int d = 0; d < D_MODEL; ++d) {
   498|        w_ext[d] = eng.byte_emit[tok_cur][d];
   499|        eps[d] = w_ext[d] - w_int[d];
   500|        err_before_sq += (double)eps[d] * (double)eps[d];
...
   580|    float holo_norm = std::sqrt((float)err_before_sq + 1e-12f);
   581|    float init_e[N_OBJ], absorbed[N_OBJ] = {}, cur_wave[N_OBJ];
   582|    for (int j = 0; j < N_OBJ; ++j) {
   583|        float dx = eng.obj_pos[j][0] - active_pos[0];
   584|        float dy = eng.obj_pos[j][1] - active_pos[1];
   585|        float dz = eng.obj_pos[j][2] - active_pos[2];
   586|        float dist = std::sqrt(dx * dx + dy * dy + dz * dz + 1e-12f);
   587|        float cos_v = std::max(0.0f, (active_norm[0] * dx + active_norm[1] * dy + active_norm[2] * dz) / dist);
   588|        init_e[j] = holo_norm * cos_v * std::exp(-dist / eng.graph.radii[j]);
   589|        cur_wave[j] = init_e[j];
   590|    }
...
   611|    float eps_tree[D_MODEL];
   612|    float rgbl_tree[4][D_MODEL];
   613|    fast_haar_forward_64_no_matrix(eps, eps_tree, cnt.mac_ops_counter);
...
   647|        for (int k = 0; k < D_MODEL; ++k) {
   648|            float rgbl_k = flt[0] * rgbl_tree[0][k] + flt[1] * rgbl_tree[1][k] + flt[2] * rgbl_tree[2][k] + flt[3] * rgbl_tree[3][k];
   649|            /* Взвешиваем на перспективную энергию узла */
   650|            tree_sig[k] = (eps_tree[k] + rgbl_k) * (0.25f + node_e);
   651|            saved_tree_sig[j][k] = tree_sig[k];
   652|        }
```

---

## 4. Все места, где 64 сектора сворачиваются в 3 числа

Во всём файле `review_package/run_task2_nanite_lod_complete.cpp` (и аналогично в `progon_holographic_relief_v8.py`, `run_T5_to_T15_suite.py` и `membrane_collision_3d.comp`) обнаружены **два места**, где 64 сектора поверхности сворачиваются в 3 числа:

1. **Сворачивание 64 позиций вершин поверхности `p_surf[64][3]` в единый 3D-центроид `active_pos[3]`** через взвешивание по ошибке `fw = (0.05f + std::fabs(eps[d])) * rad_flux[d]`:
   - `review_package/run_task2_nanite_lod_complete.cpp`, строки `523`, `566`–`568`, `575`:
     ```cpp
     523|    float active_pos[3] = {0.0f, 0.0f, 0.0f};
     ...
     566|            active_pos[0] += p_surf[d][0] * fw;
     567|            active_pos[1] += p_surf[d][1] * fw;
     568|            active_pos[2] += p_surf[d][2] * fw;
     ...
     575|    active_pos[0] /= w_sum; active_pos[1] /= w_sum; active_pos[2] /= w_sum;
     ```
2. **Сворачивание 64 нормалей секторов поверхности `(nx, ny, nz)` в единый 3D-вектор нормали `active_norm[3]`** с теми же весами ошибки `fw` (который далее используется и для всех углов `cos_v` в строке `587`, и для всех рёбер каскада `align` в строке `601`):
   - `review_package/run_task2_nanite_lod_complete.cpp`, строки `524`, `569`–`571`, `576`–`577`:
     ```cpp
     524|    float active_norm[3] = {0.0f, 0.0f, 0.0f};
     ...
     569|            active_norm[0] += nx * fw;
     570|            active_norm[1] += ny * fw;
     571|            active_norm[2] += nz * fw;
     ...
     576|    float an_len = std::sqrt(active_norm[0] * active_norm[0] + active_norm[1] * active_norm[1] + active_norm[2] * active_norm[2] + 1e-12f);
     577|    active_norm[0] /= an_len; active_norm[1] /= an_len; active_norm[2] /= an_len;
     ```

Следствие: поскольку все 64 сектора поверхности перед расчётом расстояний `dist` и углов `cos_v` сжимаются в одну точку `active_pos[3]` и одну нормаль `active_norm[3]`, локальный рельеф отдельных секторов мембраны стирается при расчёте геометрии взаимодействия с 96 объектами комнаты.
