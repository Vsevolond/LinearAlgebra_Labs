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

tuple<VectorXd, int> yakobi_method(MatrixXd A, VectorXd f, int n, double eps) {
    MatrixXd alpha(n, n);
    VectorXd betta(n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                alpha(i, j) = -A(i, j) / A(i, i);
            }
        }
        
        betta(i) = f(i) / A(i, i);
    }
    
    VectorXd x_last = betta;
    VectorXd x_curr = sum(betta, multiply(alpha, x_last, n), n); //alpha * x_last + betta;
    
    int iterations = 1;
    
    while ((x_curr - x_last).norm() >= eps) {
        x_last = x_curr;
        x_curr = sum(betta, multiply(alpha, x_last, n), n); //alpha * x_last + betta;
        
        iterations += 1;
    }
    
    return {x_curr, iterations};
}

tuple<VectorXd, int> zeidel_method(MatrixXd A, VectorXd f, int n, double eps) {
    MatrixXd alpha(n, n);
    VectorXd betta(n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                alpha(i, j) = -A(i, j) / A(i, i);
            }
        }
        
        betta(i) = f(i) / A(i, i);
    }
    
    VectorXd x_last(n);
    VectorXd x_curr = betta;
    
    int iterations = 1;
    
    while ((x_curr - x_last).norm() >= eps) {
        x_last = x_curr;
        
        for (int i = 0; i < n; i++) {
            double sum = betta(i);
            
            for (int j = 0; j < n; j++) {
                if (j < i) {
                    sum += alpha(i, j) * x_curr(j);
                    
                } else if (j > i) {
                    sum += alpha(i, j) * x_last(j);
                }
            }
            
            x_curr(i) = sum;
        }
        
        iterations += 1;
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

void test(int n, double dominance, double step) {
    vector<double> dominances;
    vector<double> yakobi_iterations;
    vector<double> zeidel_iterations;
    vector<double> yakobi_absolute_errors;
    vector<double> zeidel_absolute_errors;
    vector<double> yakobi_durations;
    vector<double> zeidel_durations;
    
    VectorXd x_correct = random_vector(n, 1000);
    double eps = 1e-6;
    
    for (double alpha = -dominance; alpha <= dominance; alpha += step) {
        MatrixXd A = random_matrix(n, alpha, 1000);
        VectorXd f = A * x_correct;
        
        auto [yakobi_result, duration_yakobi] = measure([A, f, n, eps]() {
            return yakobi_method(A, f, n, eps);
        });
        auto [x_yakobi, iter_yakobi] = yakobi_result;
        
        auto [zeidel_result, duration_zeidel] = measure([A, f, n, eps]() {
            return zeidel_method(A, f, n, eps);
        });
        auto [x_zeidel, iter_zeidel] = zeidel_result;
        
        dominances.push_back(alpha);
        yakobi_durations.push_back(duration_yakobi);
        zeidel_durations.push_back(duration_zeidel);
        yakobi_iterations.push_back(iter_yakobi);
        zeidel_iterations.push_back(iter_zeidel);
        
        double yakobi_error = (x_yakobi - x_correct).norm();
        double zeidel_error = (x_zeidel - x_correct).norm();
        
        yakobi_absolute_errors.push_back(yakobi_error);
        zeidel_absolute_errors.push_back(zeidel_error);
    }
    
    Plot2D plot1, plot2, plot3;

    plot1.xlabel("Diagonal Dominance");
    plot1.ylabel("Iterations");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();

    plot1.drawCurve(dominances, yakobi_iterations).label("Yakobi Iterations");
    plot1.drawCurve(dominances, zeidel_iterations).label("Zeidel Iterations");

    plot2.xlabel("Diagonal Dominance");
    plot2.ylabel("Duration (microseconds)");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();

    plot2.drawCurve(dominances, yakobi_durations).label("Yakobi Duration");
    plot2.drawCurve(dominances, zeidel_durations).label("Zeidel Duration");

    plot3.xlabel("Diagonal Dominance");
    plot3.ylabel("Absolute Error");
    plot3.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot3.grid().show();

    plot3.drawCurve(dominances, yakobi_absolute_errors).label("Yakobi Error");
    plot3.drawCurve(dominances, zeidel_absolute_errors).label("Zeidel Errors");

    Figure fig = {{plot1}, {plot2}, {plot3}};
    Canvas canvas = {{fig}};
    canvas.size(800, 1200);
    canvas.show();
}

int main() {
    test(10, 10, 0.5);
}
