#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <Eigen/Dense>
#include <sciplot/sciplot.hpp>
#include <chrono>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::EigenSolver;

using sciplot::Plot2D;
using sciplot::Figure;
using sciplot::Canvas;

using std::vector;
using std::function;
using std::tuple;
using std::numeric_limits;
using std::mt19937;
using std::uniform_real_distribution;
using std::random_device;
using std::swap;
using std::fmin;
using std::fmax;
using std::fabs;
using std::sqrt;
using std::srand;
using std::iter_swap;
using std::cout;
using std::endl;
using std::isnan;
using std::string;
using std::micro;

using std::chrono::duration;
using std::chrono::steady_clock;

MatrixXd random_symmetric_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    MatrixXd matrix(n, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) { matrix(i, j) = matrix(j, i); }
            else { matrix(i, j) = dist(gen); }
        }
    }
    
    return matrix;
}

MatrixXd random_symmetric_positive_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    MatrixXd matrix(n, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) { matrix(i, j) = matrix(j, i); }
            else { matrix(i, j) = dist(gen); }
            
            if (i == j) { matrix(i, j) += n * limit; }
        }
    }
    
    return matrix;
}

VectorXd random_vector(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    VectorXd vector(n);
    
    for (int i = 0; i < n; i++) {
        vector(i) = dist(gen);
    }
    
    return vector;
}

// MARK: - Классический метод Гаусса

VectorXd gauss_classic(MatrixXd A, VectorXd f, int n) {
    
    // MARK: - Прямой ход
    
    for (int k = 0; k < n - 1; k++) {
        
        // MARK: - Исключения
        
        for (int row = k + 1; row < n; row++) {
            double divider = A(row, k);
            
            for (int col = k; col < n; col++) {
                A(row, col) = A(row, col) / divider - A(k, col) / A(k, k);
            }
            
            f(row) = f(row) / divider - f(k) / A(k, k);
        }
    }
    
    // MARK: - Обратный ход
    
    VectorXd x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A(i, j) * x(j);
        }
        
        x(i) = (f(i) - sum) / A(i, i);
    }
    
    return x;
}

// MARK: - Метод Холецкого

VectorXd method_holetskovo(MatrixXd A, VectorXd f, int n) {
    
    for (int i = 0; i < n; i++) {
        double s = A(i, i);
        
        for (int k = 0; k < i; k++) {
            s = s - A(i, k) * A(i, k);
        }
        
        A(i, i) = sqrt(s);
        
        for (int j = i + 1; j < n; j++) {
            s = A(j, i);
            
            for (int k = 0; k < i; k++) {
                s = s - A(i, k) * A(j, k);
            }
            
            A(j, i) = s / A(i, i);
        }
    }
    
    VectorXd y(n);
    
    for (int i = 0; i < n; i++) {
        double sum = 0;
        
        for (int j = 0; j < i; j++) {
            sum += A(i, j) * y(j);
        }
        
        y(i) = (f(i) - sum) / A(i, i);
    }
    
    MatrixXd A_T = A.transpose();
    
    VectorXd x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A_T(i, j) * x(j);
        }
        
        x(i) = (y(i) - sum) / A_T(i, i);
    }
    
    return x;
}

template<typename Func>
auto measure(Func func) -> tuple<decltype(func()), double> {
    auto start = steady_clock::now();
    auto result = func();
    auto end = steady_clock::now();
    
    duration<double, micro> time = end - start;
    return {result, time.count()};
}

// MARK: - Test

void test(int n) {
    MatrixXd A = random_symmetric_positive_matrix(n, 1000);
    MatrixXd x_correct = random_vector(n, 1000);
    
    VectorXd f = A * x_correct;
    
    VectorXd gauss_vector = gauss_classic(A, f, n);
    
    cout << gauss_vector << endl;
    
    VectorXd holetski_vector = method_holetskovo(A, f, n);
    
    cout << holetski_vector << endl;
    
    cout << (gauss_vector - holetski_vector).norm() << endl;
}

// MARK: - Main

int main() {
    vector<double> dimensions = {3, 10, 100, 300};
    
    vector<double> gauss_durations;
    vector<double> holetski_durations;
    
    for (int dimension : dimensions) {
        int n = dimension;
        
        MatrixXd A = random_symmetric_positive_matrix(n, 1000);
        VectorXd x_correct = random_vector(n, 1000);
        
        VectorXd f = A * x_correct;
        
        auto [gauss_vector, gauss_duration] = measure([A, f, n]() {
            return gauss_classic(A, f, n);
        });
        
        auto [holetski_vector, holetski_duration] = measure([A, f, n]() {
            return method_holetskovo(A, f, n);
        });
        
        gauss_durations.push_back(gauss_duration);
        holetski_durations.push_back(holetski_duration);
    }
    
    Plot2D plot;

    plot.drawCurve(dimensions, gauss_durations).label("Gauss Classic");
    plot.drawCurve(dimensions, holetski_durations).label("Method Holetskovo");

    plot.xlabel("Matrix Dimension");
    plot.ylabel("Duration");
    plot.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot.grid().show();

    Figure fig = {{plot}};
    Canvas canvas = {{fig}};
    canvas.size(800, 600);
    canvas.show();
}
