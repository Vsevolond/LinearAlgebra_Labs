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

MatrixXd random_matrix(int n, double alpha, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    MatrixXd matrix(n, n);
    
    for (int i = 0; i < n; i++) {
        double sum = 0;
        
        for (int j = 0; j < n; j++) {
            double elem = dist(gen);
            matrix(i, j) = elem;
            
            if (j != i) { sum += fabs(elem); }
        }
        
        double elem = sum + alpha;
        matrix(i, i) = (matrix(i, i) >= 0) ? elem : -elem;
    }
    
    return matrix;
}

MatrixXd random_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    MatrixXd matrix(n, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix(i, j) = dist(gen);
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

VectorXd multiply(MatrixXd A, VectorXd v, int n) {
    VectorXd result(n);
    
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        
        for (int j = 0; j < n; j++) {
            sum += A(i, j) * v(j);
        }
        
        result(i) = sum;
    }
    
    return result;
}

VectorXd sum(VectorXd v1, VectorXd v2, int n) {
    VectorXd result(n);
    
    for (int i = 0; i < n; i++) {
        result(i) = v1(i) + v2(i);
    }
    
    return result;
}

tuple<VectorXd, int> single_parameter_method(double tau, MatrixXd A, VectorXd f, int n, double eps) {
    VectorXd x_last(n);
    VectorXd x_curr = tau * f;
    
    int iterations = 1;
    
    while ((x_curr - x_last).norm() >= eps) {
        x_last = x_curr;
        x_curr = x_last - tau * (A * x_last - f);
        
        iterations++;
    }
    
    return {x_curr, iterations};
}

template<typename Func>
auto measure(Func func) -> tuple<decltype(func()), double> {
    auto start = steady_clock::now();
    auto result = func();
    auto end = steady_clock::now();
    
    duration<double, micro> time = end - start;
    return {result, time.count()};
}

void test(int n) {
    vector<double> taus;
    vector<double> durations;
    vector<double> iterations;
    vector<double> errors;
    
    MatrixXd A = random_symmetric_positive_matrix(n, 10);
    VectorXd x_correct = random_vector(n, 10);
    VectorXd f = A * x_correct;
    
    double eps = 1e-6;
    
    EigenSolver<MatrixXd> solver(A);
    auto lambdas = solver.eigenvalues();
    
    double max_lambda = -INFINITY;
    double min_lambda = INFINITY;
    
    for (auto lambda : lambdas) {
        max_lambda = fmax(max_lambda, fabs(lambda.real()));
        min_lambda = fmin(min_lambda, fabs(lambda.real()));
    }
    
    double step = 2 / (20 * max_lambda);
    
    cout << "LAMBDA_MAX: " << max_lambda << endl;
    cout << "TAU LIMIT: " << 2 / max_lambda << endl;
    cout << "TAU OPTIMAL: " << 2 / (min_lambda + max_lambda) << endl;
    cout << "STEP: " << step << endl;
    
    for (double tau = step; tau <= 2 / max_lambda; tau += step) {
        auto [result, duration] = measure([tau, A, f, n, eps]() {
            return single_parameter_method(tau, A, f, n, eps);
        });
        
        auto [x, iter] = result;
        
        taus.push_back(tau);
        durations.push_back(duration);
        iterations.push_back(iter);
        
        double error = (x - x_correct).norm();
        errors.push_back(error);
    }
    
    Plot2D plot1, plot2, plot3;

    plot1.xlabel("Tau Parameter");
    plot1.ylabel("Duration");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();
    plot1.drawCurve(taus, durations).label("Duration");

    plot2.xlabel("Tau Parameter");
    plot2.ylabel("Iterations");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();
    plot2.drawCurve(taus, iterations).label("Iterations");

    plot3.xlabel("Tau Parameter");
    plot3.ylabel("Absolute Error");
    plot3.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot3.grid().show();
    plot3.drawCurve(taus, errors).label("Absolute Error");

    Figure fig = {{plot1}, {plot2}, {plot3}};
    Canvas canvas = {{fig}};
    canvas.size(800, 1200);
    canvas.show();
}

int main() {
    test(10);
}
