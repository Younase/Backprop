# Backprop
A MultiLayer Deep Neural Network Backpropagation implementation in C with XOR and simple digit recognition demos.


## Usage

1. Clone or download Backprop project

```
git clone https://github.com/Younase/Backprop.git
cd Backprop
```

2. compile desired demo (digit_recon.c or xor.c)

* XOR

```
gcc -Wall -o xor xor.c net.c -lm
./xor
```

* Digit Recognition

```
gcc -Wall -o digit_recon digit_recon.c net.c -lm
./digit_recon
```

## TO-DO
* ~~add support for different inputs/outputs combinations~~
* add support for saving and loading models
* return network output in feed forward function
* ~~expand readme~~
