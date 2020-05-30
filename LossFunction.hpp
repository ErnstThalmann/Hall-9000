#pragma once

// указатель на функцию
typedef double (*LossFunction)(const vector<double> &y, const vector<double> &t, vector<double> &dout);

// среднеквадратичное отклонение
double MSE(const vector<double> &y, const vector<double> &t, vector<double> &dout){
	double loss = 0; // обнуляем ошибку

	// вычисляем значение ошибки и её градиентов
	for (int i = 0; i < y.size(); i++){
		double e = y[i] - t[i];
		loss += e * e;
		dout[i] = 2 * e;
	}

	return loss; // возвращаем ошибку
}

// перекрёстная энтропия
double CrossEntropy(const vector<double> &y, const vector<double> &t, vector<double> &dout){
	double loss = 0; // обнуляем ошибку

	// вычисляем значение ошибки и её градиентов
	for (int i = 0; i < y.size(); i++){
		dout[i] = -t[i] / y[i];
		loss -= t[i] * log(y[i]);
	}

	return loss; // возвращаем ошибку
}