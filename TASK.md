# Повторение результатов статьи 

Повторяем эксперимент из статьи

```
Bunel, R., Hausknecht, M., Devlin, J., Singh, R., & Kohli, P. (2018). 
Leveraging grammar and reinforcement learning for neural program synthesis. 
arXiv preprint arXiv:1805.04276.
```

В статье предлагается использовать обучение с подкреплением и отсев генерируемых вариантов программ путем из их исполнения и проверки синтаксиса.

Используется сгенерированный датасет [karel_dataset](https://msr-redmond.github.io/karel-dataset/).

Исходный код для начала эксперимента доступен на [Github](https://github.com/bunelr/GandRL_for_NPS).

## Шпаргалка

См. подсказки по [сложным действиям](./FAQ.md)

## Как сдавать

 - [ ] решить задания по задачам, ответы на вопросы привести в этом файле или добавить в нужном пункте ссылку на ответ
 - [ ] оформить в виде git репозитория, в котором размещены все результаты
 - [ ] прислать ссылку на репозиторий письмом
 
Пример оформления ответа

 - как оформить ответ на вопрос?
> Например вот так
 - как добавить ответ ссылкой
> вот [ссылка](./TASK.md#как-сдавать)


Для обучения моделей можно использовать Google Colab c GPU.

## Задача 1. Подготовка данных (2 балла)

### Задание 1. (1 балл)

Получите karel_dataset, найдите в нем train.json и разберитесь в его структуре. 
Напишите код, который загружает данные из train.json

Вопросы
 - как правильно называется формат файла train.json?

> JSONLines - каждая строка представляет одну задачу с несколькими IO-примерами

 - как взять часть из файла train.json?

> Прочитать необходимое количество строк

 - подготовьте файлы корректного формата размером 0.1%, 1%, 3%, 10% от оригинала train.json

> Файлы тут [0.1](./data/1m_6ex_karel/train_p_0.1.json), [1](./data/1m_6ex_karel/train_p_1.json), [3](./data/1m_6ex_karel/train_p_3.json), [10](./data/1m_6ex_karel/train_p_10.json)
> ```python
> percentiles = {
>   "0_1": int(ROWS_COUNT * 0.001),
>   "1": int(ROWS_COUNT * 0.01),
>   "3": int(ROWS_COUNT * 0.03),
>   "10": int(ROWS_COUNT * 0.1),
> }
>
> for p, row_count in percentiles.items():
>   with open("./data/1m_6ex_karel/train.json", "r") as f:
>     with open("./data/1m_6ex_karel/train_p_{p}.json", "w") as f_p:
>       for _ in range(row_count):
>         f_p.write(f.readline())
> ```

 - оцените объем необходимой RAM

> Для того, чтобы поместить указанный файл в память, необходимо как минимум потратить память, пропорционально размеру файла, плюс накладные расходы python, то есть порядка нескольких десятков гигабайт, что означает, что в оперативную память это не влезет

 - реализуйте загрузку в итератор словарей (паттерн итератор) и измерьте используемый объем RAM

> ```python
> import json
>
> def get_dataset_row_iter(filename: str):
>   with open(filename, "r") as f:
>     for line in f:
>       yield json.loads(line)
>
> it = get_dataset_row_iter(f"./data/1m_6ex_karel/train.json")
> ```
>
> При использовании итератора, память будет тратиться только на 1 строку, что составляет несколько килобайт


Получите GandRL_for_NPS из github и смерджите его в этот репозиторий (см. [как это сделать](./FAQ.md#как-объединить-репозитории) в FAQ.md)
Сделайте свой клон репозитория с исправлениями. 

Вопросы
 - что нужно сделать для установки эксперимента и зависимостей?

> Необходимо добавить репозиторий с экспериментом по инструкции, провести установку согласно [README](./README.md), а также установить pytorch
> ```bash
> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
> pip3 install cython
> python3 setup.py install
> ```

 - где должны быть размещены данные?

> Данные должны быть размещены в папке [data](./data)

 - что нужно сделать для запуска эксперимента, как указать параметры и какие значения выбрать?

> Запустить эксперимент можно с помощью команды, приведенной в [README](./README.md), параметры можно указать просто передав их аргументами, значения оставим такие же, как у авторов
> ```bash
> train_cmd.py --kernel_size 3 \
>             --conv_stack "64,64,64" \
>             --fc_stack "512" \
>             --tgt_embedding_size 256 \
>             --lstm_hidden_size 256 \
>             --nb_lstm_layers 2 \
>             \
>             --signal supervised \
>             --nb_ios 5 \
>             --nb_epochs 100 \
>             --optim_alg Adam \
>             --batch_size 128 \
>             --learning_rate 1e-4 \
>             \
>             --train_file data/1m_6ex_karel/train.json \
>             --val_file data/1m_6ex_karel/val.json \
>             --vocab data/1m_6ex_karel/new_vocab.vocab \
>             --result_folder exps/supervised_use_grammar \
>             \
>             --use_grammar \
>             \
>             --use_cuda
> ```

 - что нужно сделать для проверки обученной модели?

> Авторы приводят режим Evaluation

### Задание 2. (1 балл)

Подготовьте эксперимент к запуску на Google Colab или аналогичной системе. 
Загрузите 0.1% от train.json и остальные файлы из karel_dataset.

Вопросы
 - зачем нужны файлы *.thdump в папке датасета?

> Это кеши, чтобы лишний раз не собирать датасет
> ```python3
> path_to_ds_cache = path_to_dataset.replace('.json', '.thdump')
> ```

 - что содержит new_vocab?

> Содержит токены, которые будут использованы для векторизации

 - где находится датасет для контроля и для теста?

> Датасет [train](./data/1m_6ex_karel/val.json) и [test](./data/1m_6ex_karel/test.json)

 - как устроен экземпляр данных для обучения?

Экземпляр данных для обучения состоит из:
- `guid` — id задачи
- `examples` — IO-примеры для одной программы
  - `example_index` — индекс примера внутри задачи
  - `actions` — набор действий на этом примере (как outgrid получилась из ingrid)
  - `inpgrid_json` / `outgrid_json` — человекочитаемое описание мира:
    - `rows`, `cols` — размер карты
    - `hero` — позиция и направление,
    - `blocked` — стены,
    - `markers` — маркеры,
    - `crashed` — упал ли исполнитель в этом состоянии
  - `inpgrid_tensor` / `outgrid_tensor` — описание мира, но в sparse-тензорном виде для модели ("index:1.0 index:1.0 ...")
- `program_json` — эталонная программа в AST-форме (дерево команд)
- `program_tokens` — та же программа, но токенами для seq2seq 


Проведите тестовый эксперимент с 0.1% данных и сохраните результаты обучения.

Вопросы
 - как указать вид модели?

> С помощью параметров при запуске: `--kernel_size`, `--conv_stack`, `--fc_stack`, `--tgt_embedding_size`, `--lstm_hidden_size`, `--nb_lstm_layers`

 - какие ошибки возникли при запуске и как вы их устранили?

> 1. ***TypeError: empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of:***, возникающая в ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/network.py", line 705, in __init__***
>
> 2. ***AttributeError: np.NINF was removed in the NumPy 2.0 release. Use -np.inf instead***, возникающая в ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/train.py", line 247, in train_seq2seq_model***
>
> 3. ***NameError: name 'xrange' is not defined***, возникающая в ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/data.py", line 98, in shuffle_dataset***
>
> 4. ***TypeError: Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead. Please do not rely on this error; it may not be given on all Python implementations***, возникающая в ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/karel/world.py", line 295, in fromPytorchTensor***
>
> 5. ***TypeError: conv2d() received an invalid combination of arguments - got (Tensor, Parameter, Parameter, tuple, tuple, tuple, int), but expected one of: * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, tuple of ints padding = 0, tuple of ints dilation = 1, int groups = 1) didn't match because some of the arguments have invalid types: (Tensor, Parameter, Parameter, tuple of (int, int), tuple of (float, float), tuple of (int, int), int)***, возникающая в ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/network.py", line 58, in forward***
>
> 6. ***RuntimeError: masked_fill only supports boolean masks, but got dtype Byte***, возникающая в ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/network.py", line 268, in forward***
>
> 7. ***IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number***, возникающая в файле ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/training_functions.py", line 30, in do_supervised_minibatch***
>
> 8. ***_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source. (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message. WeightsUnpickler error: Unsupported global: GLOBAL nps.network.IOs2Seq was not an allowed global by default. Please use `torch.serialization.add_safe_globals([nps.network.IOs2Seq])` or the `torch.serialization.safe_globals([nps.network.IOs2Seq])` context manager to allowlist this global if you trust this class/function***, возникающая в ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/evaluate.py", line 90, in evaluate_model***
>
> 9. ***ValueError: too many values to unpack (expected 2)***, возникающая в файле ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/data.py", line 141, in get_minibatch***
>
> 10. ***TypeError: only integer tensors of a single element can be converted to an index***, возникающая в файле ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/network.py", line 358, in beam_sample***
>
> 11. ***TypeError: list indices must be integers or slices, not float***, возникающая в файле ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/nps/network.py", line 358, in beam_sample***
>
> 12. ***AttributeError: 'dict' object has no attribute 'has_key', возникающая в файле ***File "/home/xbloody/.venv/lib/python3.12/site-packages/G_RLForNPS-0.0.1.dev1-py3.12-linux-x86_64.egg/karel/ast.py", line 10, in __init__***

 - сколько эпох вы провели?

> Провел 10 эпох, потому что данных сильно меньше

 - где сохранены результаты и логи эксперимента?

> В файле [logs.txt](./exps/supervised_use_grammar/logs.txt)

 - какого качества получен результат?

 > Accuracy низкое (0) и лосс достаточно большой (1.11384). Качество низкое из-за очень маленького датасета

Подсказки

2а. Оператор m += 1 имеет другое значение, чем m = m + 1. Также воспользуйтесь явным приведением типов
 
2б. Из-за разных версий torch может потребоваться адаптировать код проекта. Например, для получения значения из тензора размером 1 нужно вызывать ``item()`` вместо ``[0]``

2в. Удалите кэши данных перед запуском

2г. Попробуйте заменить tqdm на progressbar2, см. [вот это сообщение](https://github.com/tqdm/tqdm/issues/613) или установить версию 2018 года.

2д. После обновления кода не забывайте установить повторно через setup.py


## Задача 2. Подготовка репозитория (3 баллов)

### Задание 2. (1 балл)

Изучите алгоритмы обучения разных видов моделей. 

Вопросы
 - какие алгоритмы обучения реализованы в проекте и в каких модулях и функциях?
 - каким образом реализована проверка правильности синтаксиса генерируемой программы?



 - где в коде формируются минибатчи для обучения?

В файле [nps/data.py](./nps/data.py) в функции `get_minibatch`. Она принимает на вход стартовый индекс (курсор) и размер батча, после чего формирует из датасета необходимые структуры

- `inp_grids` - батч тензоров-входных карт (трехмерный тензор, где размерность каналов кодирует положение героя, направление взгляда, препятствия и маяки в виде OneHot)

- `inp_grids` - батч тензоров-выходных карт в аналогичном формате

- `in_tgt_seq` - батч наборов токенов на вход модели (в виде индексов в словаре), выравненный паддингом по максимальной длине

- `input_lines` - батч тех же наборов токенов, только в виде `list[list[int]]`, а не в виде тензора

- `out_tgt_seq` - батч тех же наборов токенов, сдвинутый на один влево (для i-го входа это эталонное предсказание)

- `inp_worlds` - набор входных миров Karel (нужны для симуляции в RL)

- `out_worlds` - набор выходных (эталонных) миров Karel (нужны для Consistency сред - сравниваем inp и out)

- `targets` - батч эталонных программ в виде индексов токенов

- `inp_test_worlds` - набор входных "тестовых" миров Karel

- `out_test_worlds` - набор выходных "тестовых" миров Karel (нужны для Generalization/Perf сред, где поощряется корректная работа на примерах не из обучающей выборки)

 - какой критерий (функция потерь) используется при обучении с учителем и зачем в ней веса?

> При обучении с учителем использует Кросс-Энтропийная функция потерь. На вход принимает логиты сети и применяет к ним `softmax` 
> 
> $$p_c^{(n)} = \frac{\exp\left(x_c^{(n)}\right)}{\sum_{c=0}^C\exp\left(x_c^{(n)}\right)}$$
> 
> $$\mathcal{l}_n = -\sum_{c=1}^C \omega_c y_{n, c} \log p_{n, c} = -\omega_{y_n}\log p_{n, y_n}$$
> 
> Веса для всех токенов, кроме токена паддинга задаются 1, для паддинга - 0 для того, чтобы не штрафовать модель за ошибки с паддингом (он не влияет на работоспособность программы)

 - в чем состоит алгоритм reinforce?
 - поясните различие между RL за два шага и за один шаг с разверткой (rollout)

### Задание 3. (1 балл)

Подготовьте и добавьте архивы подготовленных данных 1%, 3%, 10% на файлообменник с доступом по прямой ссылке
 
Вопросы
 - как получить данные по ссылке из командной строки?

> Архивы подготовленных данных находятся на [Google Drive](https://drive.google.com/drive/folders/1DqBI45BLlUJ4U6fG66uyAXTrxVqlxTbQ?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)
> Скачать каждый файл можно с помощью утилиты `gdown`
> ```bash
> gdown 1pLMovHpgKTRYdSgcuVtrNvPpbNayC7kE # 10%
> gdown 1aDgsnR3cG1z9Vavdfclcer_0ENDwj4Jk # 3 %
> gdown 1nJkJdw-L7q3ErqYLUHLM_WghX4qd_rH3 # 1 %
> ```

 - где находится ваш репозиторий?

> Репозиторий находится на [GitHub](https://github.com/pavelshapka/auto_karel_synthesis)

 - исходя из результатов на 0.1% данных оцените время обучения на 1%, 3%, 10% данных



 - убедитесь, что обученные промежуточные модели сохраняются в надежном месте по ходу обучения и объясните как это работает

Сохраняется тут // TODO где?

При прерывании обучения и повторном запуске (если передать параметр `init_weights`) в строках `nps/train.py:205-206` происходит загрузка модели из файла `.model`.

### Задание 4. (1 балл)

Подключите журналирование кривых обучения и промежуточных результатов в [TensorboardX](https://github.com/lanpa/tensorboardX). 
Для этого нужно использовать SummaryWriter для сохранения промежуточных значений функции потерь внутри цикла обучения. Например, после каждого минибатча или 100 миниматчей. 

Логи сохранять в подпапку ``runs`` папки эксперимента в ``./exps/*``

Для использования установить Tensorboard и указать данную папку в качестве исходной.
  
## Задача 3. Анализ и повторение результата (5 балла)

### Задание 1. (2 балла)

Проведите эксперименты для 1%, 3%, 10% данных для MLE (supervised), RL_beam и обучаемой и предзаданной моделью синтаксиса

Вопросы
 - сколько времени заняло проведение эксперимента, какие ресурсы использовали?
 - какого качества удалось достичь?
 - приведите 10 примеров синтеза программы для karel обученной моделью

Начните с MLE и используйте обученную таким образом модель для RL в качестве начального приближения.
См. подробнее в репозитории проекта.

### Задание 2. (3 балла)

Подготовьте слайды по эксперименту: 
1. название статьи и постановка задачи (по статье, формулы) 
2. использованные данные (ссылка и как готовили)
3. постановка эксперимента (репо и команды запуска)
4. таблица с результатами (сводная)

Вопросы
 - сравните с результатами для Small dataset из статьи, удалось ли повторить их результаты?
 - как влияет выбор алгоритма контроля корректности программы на качество, в зависимости от размера?


