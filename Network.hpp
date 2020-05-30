#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

#include "Layers/Layer.hpp"
#include "Layers/FullConnectedLayer.hpp"
#include "Layers/ActivationLayer.hpp"
#include "Layers/SoftmaxLayer.hpp"
#include "LossFunction.hpp"

using namespace std;

class Network{
    int inputSize, outputSize; // количество входов и выходов
    vector<Layer*> layers; // слои

    void Backward(const vector<double> &x, const vector<double> &dout); // обратное распространение
    int Argmax(const vector<double> &v) const; // индекс максимума
public:
    Network(int inputSize); // создание нейронной сети из размера

    void AddLayer(const string& description); // добавление слоя
    vector<double> Forward(const vector<double> &x); // прямое распространение
    
    void Train(const Data &trainData, const Data& testData, double learningRate, int epochs, LossFunction L, double eps); // обучение сети
    double Test(const Data &data); // проверка точности
    void Summary() const; // вывод информации о сети
};

// обратное распространение
void Network::Backward(const vector<double> &x, const vector<double> &dout) {
    if (layers.size() == 1) { // если слой один
        layers[layers.size() - 1]->Backward(x, dout, false); // делаем обратное распространение на нём
        return;
    }

    layers[layers.size() - 1]->Backward(layers[layers.size() - 2]->output, dout, true); // обратно распространяем последний слой

    for (int i = layers.size() - 2; i >= 1; i--)
        layers[i]->Backward(layers[i - 1]->output, layers[i + 1]->dx, true); // обратно распространяем остальные слои

    layers[0]->Backward(x, layers[1]->dx, false); // обратно распространяем первый слой
}

// создание нейронной сети из размера
Network::Network(int inputSize) {
    this->inputSize = inputSize;
    outputSize = inputSize;
}

// добавление слоя
void Network::AddLayer(const string& description) {
    stringstream ss(description); // создаём поток из конфигурации
    string name;
    ss >> name; // считываем название слоя
    
    if (name == "fc" || name == "fullconnected") {
        int size;
        ss >> size; // считываем размер слоя

        layers.push_back(new FullConnectedLayer(outputSize, size));
        outputSize = size;
    }
    else if (name == "activation") {
        string function;
        ss >> function; // считываем функцию активации

        layers.push_back(new ActivationLayer(outputSize, function));
    }
    else if (name == "softmax") {
        layers.push_back(new SoftmaxLayer(outputSize));
    }
    else
        throw runtime_error("Unknown layer: " + name);
}

// прямое распространение
vector<double> Network::Forward(const vector<double> &x) {
    layers[0]->Forward(x);

    for (int i = 1; i < layers.size(); i++)
        layers[i]->Forward(layers[i - 1]->output);

    return layers[layers.size() - 1]->output;
}

// индекс максимума
int Network::Argmax(const vector<double> &v) const {
    int imax = 0;

    for (int i = 1; i < v.size(); i++)
        if (v[i] > v[imax]) // если число стало больше
            imax = i; // обновляем индекс максимума

    return imax; // возвращаем индекс максимума
}

// проверка точности
double Network::Test(const Data &data) {
    double correct = 0; // обнуляем счётчик корректных

    for (int i = 0; i < data.x.size(); i++)
        if (Argmax(Forward(data.x[i])) == Argmax(data.y[i])) // если индексы максимумов совпали
            correct++; // увеличиваем количество корректных

    return correct / data.x.size(); // возвращаем точность
}

// обучение сети
void Network::Train(const Data &trainData, const Data& testData, double learningRate, int epochs, LossFunction L, double eps) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double loss = 0; // обнуляем ошибку
    
        for (int i = 0; i < trainData.x.size(); i++) {
            vector<double> out = Forward(trainData.x[i]); // прямое распространение
            vector<double> dout(outputSize); // создаём дельты

            loss += L(out, trainData.y[i], dout); // считаем ошибку
            Backward(trainData.x[i], dout); // обратное распространение
            
            // обновляем веса
            for (int j = 0; j < layers.size(); j++)
                layers[j]->UpdateWeights(learningRate);
        }
        
        loss /= trainData.x.size(); // усредняем ошибку
        cout << "Epoch: " << epoch << ", loss: " << loss << ", test_accuracy: " << Test(testData) << endl; // выводим информацию об эпохе

        if (loss < eps) // если достигли заданной ошибки
            break; // выходим
    }
}

// вывод информации о сети
void Network::Summary() const {
    cout << "+---------------------+---------------+----------------+--------------+" << endl;
    cout << "|     layer name      |  inputs count |  outputs count | weghts count |" << endl;
    cout << "+---------------------+---------------+----------------+--------------+" << endl;

    for (int i = 0; i < layers.size(); i++)
        layers[i]->Summary(); // выводим информацию о каждом слое

    cout << "+---------------------+---------------+----------------+--------------+" << endl;
}