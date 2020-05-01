1. Как запускать
В Windows 7:
Нужно скопилировать blurer.cu в blurer.dll:
nvcc -o blurer.dll --shared -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64" -I"C:\Program Files (x86)\Windows Kits\8.1\Include\um" -I"./" -L"C:\Program Files (x86)\Windows Kits\8.1\Lib\winv6.3\um\x64" -I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.10240.0\ucrt" -L"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10240.0\ucrt\x64" blurer.cu
Возможно параметры nvcc будут другие в зависимости от версии виндос, версии nvcc, версии Visual Studio
Если у вас виндос можно попробовать не компилирвать, заюзать туже dll которую я приложил
Затем запустить в python3 файл blurer_test_speed.py
Будет запущен тест на скорость в котором одно и то-же изображение блюрится 1000 раз.
В результате будет распечатано среднее время на 1 блюр и сохранен результат блюра в файлы:
img_output_cuda_test_speed.png
img_output_cuda_test_speed.npy
В blurer_test_speed.py есть класс Blurer его его можно скопипастить в свои тестовые скрипты и там уже проверять.
В Линукс:
в Линукс я не проверял, но старался писать кросплатформенные коды.
Наверное команда компиляции в линукс будет такая:
nvcc -Xcompiler -fPIC -shared -o blurer.so blurer.cu
(как в примере: https://github.com/sesutton93/C-Types-Cuda)
возможно понадобиться добавить еще какие то флаги, я не проверял.

2. Результат теста на скорость
0.02513846206665039 сек в среднем на одно (трехканальное) изображение 1200x800
Это чисто на блюр. Время на инициализации и чтение/запись файлов не включены.
Время на предачу изображения из CPU памяти в GPU память и обратно включено.
Тестировалось на железе:
CPU: Intel Core 2 Quad Q8400
GPU: NVIDIA GeForce GTX 760, 2 GB
В процессе работы GPU грузилось на 73% и использовалось 35 МБ видеопамяти.

3. Описание класса Blurer в blurer_test_speed.py
Предполагается что Blurer будет блюрить много изображений одинакового размера и блюр будет с одинаковым ядром.
Это сделано для скорости.
Если нужно менять размеры, то придется делать новый инстанс Blurer с новыми размерами (что занимает некоторое время)
Blurer инкапсулирует в себе shared library blurer.dll (является оберткой).
Даже иницализацию сам не делает а вызывает соответсвующую функцию внутри blurer.dll.

4. Как распаралеливается
CUDA - сама распаралеливает. 
см blurKernel в blurer.cu 
это kernel который запускаеться много раз паралельно на GPU.
Сетка задается по выходу. Для заданного пикселя + заданого одного из каналов RGB на выходном изображении делается 1 запуск blurKernel
Используется целочисленная арифметика (для скорости),
ядро свертки задается unsigned short int (16 бит) и значения пикселей в изображении во время расчета тоже переводится в unsigned short int
После вычисления делится на 256 и преводится обратно в unsigned cher (8 бит).
Получается что ядро свертки в пониженной точности считалось (от 0 до 255) не float, возможно поэтому результат блюра чуть-чуть различается.
Статистика этого различия:
min = -5
max = 7
mean =  0.7363226188483967
std = 0.6413344010889628
Также ядро свертики хранится в CUDA-константе для скорости, см __constant__ unsigned short core_gpu в blurer.cu
причем размер маcсива констант задается заранее как 64*64 поэтому есть ограничение на максимальный размер ядра свертки в 64x64
Можно попробовать еще ускорить - использовать __shared__ но я не делал, так как это долго,
но я умею это делать, см мой проект: https://github.com/vedenev/cpp_cuda_speedup_lenet

5. История разработки
Все промежуточные сырые коды которые я делал в процессе поиска решения находятся в отдельной папке research, см описание в конце
Поиск правильных ключей для команды nvcc занял довольно много времнии.
Я смотрел как выглядит командная строка в Visual Studio для уже готовых проектов из CUDA Samples и брал от туда некоторые флаги и пути.
Описание файлов в папке reserch:
blur.cu - первый рабочий C++/CUDA код без класса, чисто на функиях, который делает блюр через вызов из питона, см. t004_test_ctypes_2.py
blur.dll - компилированный blur.cu
blurer.cu - C++/CUDA код с классом и адаптацией для ctype, который делает блюр через вызов из питона, см. t006_test_blurer_dll.py
blurer.dll - компилированный blurer.cu
commands_tmp2.txt - поиск команды копмиляции на виндос, последняя команда рабочая
helper_cuda.h, helper_string.h - вспомогательные файлы для blur.cu и blurer.cu которые я вытащил из CUDA Samples/v8.0/common/inc
img.jpg - оригиналное тестове изображение со змеей, 1200x800
img_output.png, img_output.npy - результат блюра оригинального python/pytorch кода, см. t002_test_task_corrected.py
img_output_cuda.png, img_output_cuda.png - результат блюра t004_test_ctypes_2.py
img_output_cuda_2.png, img_output_cuda_2.png - результат блюра t006_test_blurer_dll.py
pytorch_vs_cuda_results_difference.png - график: поэлементное сравение блюров: python-pytorch и python-dll-CUDA (t002_test_task_corrected.py vs t004_test_ctypes_2.py) (есть небольшая разница)
t001.py - тест pytorch
t002_test_task_corrected.py - модифицированный оригинальный пайторчевский код test_task.py под более старую верисию пайторча: torch==1.0.0 torchvision==0.2.1
t003_test_ctypes.py - запуск тестовго примера (поэлементное возведение в квадрат) про ctypes+CUDA: https://medium.com/@akshathvarugeese/cuda-c-functions-in-python-through-dll-and-ctypes-for-windows-os-c29f56361089 
t004_numpy_flatten_learn.py - изучение .flatten(), поиск правильного порядка индексов
t004_test_ctypes_2.py - делает python-dll-CUDA блер на GPU без классов, через функции, использует blur.dll
t005_compare_cuda_and_pytorch.py - сравнивает результаты блеров t002_test_task_corrected.py vs t004_test_ctypes_2.py (есть небольшая разница)
t006_test_blurer_dll.py - делает python-dll-CUDA блер на GPU с классом, использует blurer.dll
test_task.pdf - текст тестовго задания
test_task.py - оригинальный python-pytorch скрипт который делает блер
