## CUDA blur image  
test task solved  
Interviewer allowed me to put my solution into public access  
 
  
## 1. test task definition:  
Let we have Python+Pytorch code (see research/test_task.py) that makes image blur. The task is to make CPU or GPU code that do the same.  
Also additional requirements:  
- the code must be callable from python  
- must be as fast as possible  
  
## 2. How to run  
Windows 7:  
It is need to compile blurer.cu to blurer.dll:  
```nvcc -o blurer.dll --shared -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64" -I"C:\Program Files (x86)\Windows Kits\8.1\Include\um" -I"./" -L"C:\Program Files (x86)\Windows Kits\8.1\Lib\winv6.3\um\x64" -I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.10240.0\ucrt" -L"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10240.0\ucrt\x64" blurer.cu```  
Probably parameters for nvcc will be different depending on Windows version, nvcc version, Visual Studio version  
If you are using Windows try to use compiled blurer.dll in the repo  
Then run in python3 file blurer_test_speed.py  
It will make blurr of an image 1000 times.  
It will print meant time for 1 blur and will save blur result image in to file:  
img_output_cuda_test_speed.png  
img_output_cuda_test_speed.npy  
There class Blurer In blurer_test_speed.py you can use it on your own.  
In Linux:  
I didn't check it in Linux but I tried to write cross-platform code.  
Probably compilation comand will be following:  
```nvcc -Xcompiler -fPIC -shared -o blurer.so blurer.cu```   
(like in example: https://github.com/sesutton93/C-Types-Cuda)  
Probably it is need to add more paramaters, I did't check.  
  
## 2. Resuts of speed test  
0.02513846206665039 sec as mean for one (RGB) image 1200x800  
It is only time for blur, time for intializations and for files reading/writing was not included.  
Time for transfer image from CPU memory to GPU memory and back was included.  
It was tested with hardware:  
CPU: Intel Core 2 Quad Q8400  
GPU: NVIDIA GeForce GTX 760, 2 GB  
GPU load was 73% and It was used 35 MB of GPU memory.  
  
## 3. Description of class Blurer in blurer_test_speed.py  
class Blurer are build for processing a lot of images with same sizes and with fized size of blur kernel.  
It is need for speed.  
If it is need to change sizes, then it is need to make new instance of the class Blurer.  
class Blurer uses shared library blurer.dll inside (it is binding)  
The binding is done with ctypes (python library for bind C function with Python)  
It even does not do intialization by itself, it call special function in blurer.dll.  
  
## 4. How it parallelize  
CUDA do this.  
look blurKernel in blurer.cu  
It kernel that are run many times on GPU in parallel.  
The grid are defined by output. There is 1 run of blurKernel for a fixed pixel and fixed R G or B in output image.  
It uses integer arithmetics (for speed),  
convolution kernel has unsigned short int (16 bits) precission and input image pixel values are also converted to unsigned short int (16 bits) in calculation.  
The result after calculation is devided by 256 and converted back to unsigned char (8 bits).  
So convolution kernel is calculated with low precission from 0 to 255, not float32 like in pytorch code. So results a little different.  
Statistics of this difference:  
min = -5  
max = 7  
mean =  0.7363226188483967  
std = 0.6413344010889628  
Also convolution kernel values are stored in constant array, look ```__constant__``` unsigned short core_gpu в blurer.cu, it is for speed.  
Also the maixmal size of the constant array is 64x64, so maximal size of the convolution kernel is 64x64  
It is possible to accelerate more with ```__shared__``` but I did't do it.  
I used ```__shared__``` in my another project: https://github.com/vedenev/cpp_cuda_speedup_lenet  
  
## 5. History of development  
All intermidiate raw codes a placed in to research, look details at the end of this document.  
Search of correct parameters for nvcc took a lot of time.  
I was looking at command line structure in Visual Studio in ready projects from the CUDA Samples and use it as example.   
Description of files in reserch folder:  
blur.cu - forst workable C++/CUDA code without classes, only functions, it do blur by call from python, look t004_test_ctypes_2.py  
blur.dll - compiled blur.cu  
blurer.cu - C++/CUDA code with class and with adapter for class via ctypes, it does blur with call from python, see t006_test_blurer_dll.py  
blurer.dll - compiled blurer.cu  
commands_tmp2.txt - list of command for windows, last command is workable.  
helper_cuda.h, helper_string.h - auxiliary files for blur.cu and blurer.cu that I got from CUDA Samples/v8.0/common/inc  
img.jpg - original test image with snake, 1200x800  
img_output.png, img_output.npy - result blured image of original python/pytorch code, see t002_test_task_corrected.py  
img_output_cuda.png, img_output_cuda.png - result of blur of t004_test_ctypes_2.py  
img_output_cuda_2.png, img_output_cuda_2.png - result of blur of t006_test_blurer_dll.py  
pytorch_vs_cuda_results_difference.png - graph: elementwise comparizon of blur results: python-pytorch vs python-dll-CUDA (t002_test_task_corrected.py vs t004_test_ctypes_2.py) (small difference exists)  
t001.py - test of pytorch  
t002_test_task_corrected.py - modified original pytorch code test_task.py for more old version of the pytorch: torch==1.0.0 torchvision==0.2.1  
t003_test_ctypes.py - run of test example (elemntwise square), it is about ctypes+CUDA: https://medium.com/@akshathvarugeese/cuda-c-functions-in-python-through-dll-and-ctypes-for-windows-os-c29f56361089   
t004_numpy_flatten_learn.py - learning of .flatten(), search for right indexes order  
t004_test_ctypes_2.py - does python-dll-CUDA blur on GPU without classes, only functions, uses blur.dll  
t005_compare_cuda_and_pytorch.py - comapre results of blurs t002_test_task_corrected.py vs t004_test_ctypes_2.py (small difference exists)  
t006_test_blurer_dll.py - does python-dll-CUDA blur on GPU with class, uses blurer.dll  
test_task.pdf - text of the test task in russian  
test_task.py - original python-pytorch scipt thet does blur  
