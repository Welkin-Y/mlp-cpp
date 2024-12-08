
# README

- Project adapted from [Math 49114 Scientific computing](<https://www.cl.eps.manchester.ac.uk/medialand/maths/units/2014-2015/Scientific%20Computing%20(MATH49111).pdf>)
- A single multilayer perceptron (MLP) neural network implemented by c++

# Usage

- Setup

```shell
make all
./executable <path/to/traindata> </path/to/output> \
</path/to/log> <learning_rate> <cost_tolerance> <max_iteration>
```

- Tear Down

```shell
make clean
```

# TODO

- [x] Template class
- [ ] CPU parallel version
- [ ] GPU version
