# FHE Decision Tree Train & Predict
## About the repo
This repository utilizes Fully Homomorphic Encryption (FHE) to enable machine learning over encrypted data. In this code we implement a new privacy-preserving solution to training and prediction for trees, proposed by A. Akavia, M. Leibovich, Y. S. Resheff,
R. Ron, M. Shahar, and M. Vald [1].

###### The code uses the CKKS implementation by [SEAL](https://github.com/microsoft/SEAL), an open source library provided by Microsoft.

### Dependencies
1. First, update and install c++ tools:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt install build-essential cmake g++
```
2. Install and build SEAL library (version 4.0.0)
```
git --branch 4.0.0 https://github.com/Microsoft/SEAL.git
cd SEAL
cmake -S . -B build
cmake --build build
sudo cmake --install build
```
Make sure that the SEAL installation path matches a path listed in CMakeLists.txt under the find_package() command

### Building
To build the project, run the following commands:
```
cd fhe_random_forest
cmake -S . -B build
cmake --build build
```
This command will create a 'build' subdirectory and store the executables in it. 


### Testing
After each change to the code, make sure the tests still run successfully:
```
cd build
ctest -C
```

### Running
After building and passing the tests, you can run the code example:
```
cd build
./fhe_random_forest
```

### Using Docker
You can also use the provided Dockerfile to build the code with its dependencies on Docker, then run the code example and the tests
```
cd fhe_random_forest
docker build -t <image name> .
docker run <image name>
docker run <image name> ctest -C
```

### Acknowledgment
We would like to thank all the code contributors to this project:\
Max Leibovich, github: https://github.com/smerte  
Omer Sadeh, github: https://github.com/Omer-Sadeh  
Boaz Sapir, github: https://github.com/boazsapir  
Margarita Vald, email: margarita.vald@cs.tau.ac.il

[1] A. Akavia, M. Leibovich, Y. S. Resheff, R. Ron, M. Shahar, and M. Vald. 2022. Privacy-Preserving Decision Trees Training and Prediction. ACM Trans. Priv. Secur. 25, 3, Article 24 (August 2022), 30 pages. https://doi.org/10.1145/3517197