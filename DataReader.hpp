#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

// структура для обучающих данных
struct Data {
	vector<vector<double>> x;
	vector<vector<double>> y;
};

class DataReader{
	int width, height; // размеры картинки
	vector<string> labels; // метки

	vector<string> SplitLine(string line, char separator) const; // разбиение по символу
public:
	DataReader(const string &path);
	
	int GetSize() const; // получение размера векторов
	vector<double> PixelsToVector(const vector<string> &values) const;
	vector<double> LabelToVector(const string &label) const;
	Data ReadData(const string &path);
};

DataReader::DataReader(const string &path) {
	ifstream f(path);

	if (!f)
		throw string("Unable to open file '") + path + "'"; //вывод информации об ошибке
	
	string line;
	getline(f, line); // считываем строку с размерами
	vector<string> s = SplitLine(line, ' '); // разбиваем строку с размерами по пробелу
	
	width = stoi(s[0]); // получаем ширину
	height = stoi(s[1]); // получаем высоту

	getline(f, line); // считываем строку с классами
	labels = SplitLine(line, ' '); // разбиваем строку по пробелам для получения классов
}

// разбиение строки по разделителю
vector<string> DataReader::SplitLine(string line, char separator) const {
	vector<string> s;
	string l = "";
	
	for (int i = 0; i < line.length(); i++){
		if (line[i] == separator) { // если разделитель
			s.push_back(l); // добавляем строку в массив
			l = ""; // сбрасываем строку
		}
		else
		 	l += line[i]; // добавляем символ
	}

	if (l != "") // если что-то есть
		s.push_back(l); // добавляем в массив

	return s; // возвращаем массив строк
}

// получение размера векторов
int DataReader::GetSize() const {
	return width * height;
}

// получение вектора пикселей
vector<double> DataReader::PixelsToVector(const vector<string> &values) const {
	vector<double> res(width * height); // создаём вектор

	for (int i = 0; i < height * width; i++)
		res[i] = atoi(values[i + 1].c_str()) / 255.0; // переводим в числовой вид и делим на 255

	return res; // возвращаем вектор
}

// получение вектора для метки
vector<double> DataReader::LabelToVector(const string &label) const {
	vector<double> res(labels.size());

	for (int i = 0; i < labels.size(); i++)
		if (labels[i] == label) // если нашли метку
			res[i] = 1; // проставляем 1

	return res; // возвращаем вектор
}

// считывание данных из файла
Data DataReader::ReadData(const string &path) {
	Data data;
	ifstream f(path); // открываем файл

	string line;
	getline(f, line); // пропускаем первую строку

	while (getline(f, line)) { // считываем строки из файла
		vector<string> s = SplitLine(line, ','); // разбиваем их по запятой

		data.x.push_back(PixelsToVector(s)); // добавляем изображение
		data.y.push_back(LabelToVector(s[0])); // добавляем метку
	}

	return data; // возвращаем данные
}