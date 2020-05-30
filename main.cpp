#include <iostream>
#include "DataReader.hpp"
#include "Network.hpp"

using namespace std;

int main() {
    double learningRate = 0.08; // скорость обучения
    int epochs = 500; // эпохи
	cout << "Hello, Konstantin Konstantinovich..." << endl<<endl;
    DataReader reader("dataset/mnist.txt"); // создаём считыватель данных
    Data dataTrain = reader.ReadData("dataset/mnist_train.csv");
    Data dataTest = reader.ReadData("dataset/mnist_test.csv");
        
    // выводим размеры считанных данных
    cout << "Read " << dataTrain.x.size() << " train objects" << endl;
    cout << "Read " << dataTest.x.size() << " test objects" << endl;

    // создаём сеть и добавляем в неё слои
    Network network(reader.GetSize());
    network.AddLayer("fc 64");
    network.AddLayer("activation sigmoid");
    network.AddLayer("fc 10");
    network.AddLayer("softmax");

    network.Summary(); // выводим информацию о сети

    network.Train(dataTrain, dataTest, learningRate, epochs, CrossEntropy, 1e-5); // обучаем сеть
}