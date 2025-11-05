#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using Eigen::MatrixXd;
using Eigen::VectorXd;

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

void swap(VectorXd &v, int i, int j) {
    if (i != j) {
        std::swap(v(i), v(j));
    }
}

void swap_row(MatrixXd &A, int i, int j) {
    if (i != j) {
        A.row(i).swap(A.row(j));
    }
}

void swap_col(MatrixXd &A, int i, int j) {
    if (i != j) {
        A.col(i).swap(A.col(j));
    }
}

VectorXd gauss_combined(MatrixXd A, VectorXd f, int n) {
    
    double max_0 = A(0,0);
    
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            double elem = fabs(A(row, col));
            if (elem > max_0) { max_0 = elem; }
        }
    }
    
    VectorXd permutations = VectorXd::LinSpaced(n, 0, n - 1);
    
    for (int k = 0; k < n - 1; k++) {
        
        int row_max = k;
        int col_max = k;
        
        for (int row = k; row < n; row++) {
            for (int col = k; col < n; col++) {
                double curr_elem = fabs(A(row, col));
                double max_elem = fabs(A(row_max, col_max));
                
                if (curr_elem > max_elem) {
                    row_max = row;
                    col_max = col;
                }
            }
        }
        
        swap_row(A, k, row_max);
        swap_col(A, k, col_max);
        
        swap(f, k, row_max);
        swap(permutations, k, col_max);
        
        for (int row = k + 1; row < n; row++) {
            double divider = A(row, k);
            
            for (int col = k; col < n; col++) {
                A(row, col) = A(row, col) / divider - A(k, col) / A(k, k);
            }
            
            f(row) = f(row) / divider - f(k) / A(k, k);
        }
    }
    
    VectorXd x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A(i, j) * x(j);
        }
        
        x(i) = (f(i) - sum) / A(i, i);
    }
    
    VectorXd result(n);
    
    for (int i = 0; i < n; i++) {
        int index = static_cast<int>(permutations(i));
        result(index) = x(i);
    }
    
    return result;
}

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

MatrixXd similarity_matrix(MatrixXd D, int n, int col) {
    MatrixXd B(n, n);
    
    for (int j = 0; j < n; j++) {
        if (j == col) {
            B(col, j) = 1 / D(col + 1, col);
            
        } else {
            B(j, j) = 1;
            B(col, j) = -D(col + 1, j) / D(col + 1, col);
        }
    }
    
    return B;
}

MatrixXd similarity_inverted_matrix(MatrixXd D, int n, int col) {
    MatrixXd B_inverted(n, n);
    
    for (int j = 0; j < n; j++) {
        B_inverted(col, j) = D(col + 1, j);
        
        if (j != col) {
            B_inverted(j, j) = 1;
        }
    }
    
    return B_inverted;
}

tuple<MatrixXd, MatrixXd> frobenius_matrix(MatrixXd A, int n) {
    MatrixXd D = A;
    MatrixXd B_res = MatrixXd::Identity(n, n);
    
    for (int col = n - 2; col >= 0; col--) {
        MatrixXd B = similarity_matrix(D, n, col);
        B_res = B_res * B;
        
        MatrixXd C = D * B;
        MatrixXd B_inverted = similarity_inverted_matrix(D, n, col);
        
        D = B_inverted * C;
    }
    
    return {D, B_res};
}

function<double(double)> characteristic_equation_builder(VectorXd coefs, int n) {
    return [coefs, n](double x) {
        double res = pow(x, n);
        
        for (int i = 0; i < n; i++) {
            res -= pow(x, n - i - 1) * coefs[i];
        }
        
        return res;
    };
}

function<double(double)> characteristic_equation_derivative_builder(VectorXd coefs, int n) {
    return [coefs, n](double x) {
        double res = n * pow(x, n - 1);
        
        for (int i = 0; i < n - 1; i++) {
            res -= (n - i - 1) * pow(x, n - i - 2) * coefs[i];
        }
        
        return res;
    };
}


vector<tuple<double, double>> gershgorin_circles(MatrixXd A, int n) {
    vector<tuple<double, double>> circles(n);
    
    for (int i = 0; i < n; i++) {
        double center = A(i, i);
        double radius = 0;
        
        for (int j = 0; j < n; j++) {
            if (j != i) { radius += fabs(A(i, j)); }
        }
        
        double left = center - radius;
        double right = center + radius;
        
        circles[i] = {left, right};
    }
    
    return circles;
}

double newton_single_root(function<double(double)> f, function<double(double)> df, double x0, double eps = 1e-6) {
    double x = x0;

    while (true) {
        double y = f(x);
        double dy = df(x);

        if (fabs(dy) < 1e-12) {
            break;
        }

        double x_new = x - y / dy;

        if (fabs(x_new - x) < eps) {
            return x_new;
        }

        x = x_new;
    }

    return NAN;
}

vector<double> newton_roots(
                            function<double(double)> f,
                            function<double(double)> df,
                            double start,
                            double end,
                            double step = 0.1,
                            double eps=1e-6
                            )
{
    vector<double> roots;

    for (double x0 = start; x0 <= end; x0 += step) {
        double root = newton_single_root(f, df, x0, eps);

        if (!isnan(root)) {
            bool is_new = true;
            
            for (double r : roots) {
                if (fabs(r - root) < eps) {
                    is_new = false;
                    break;
                }
            }
            
            if (is_new) roots.push_back(root);
        }
    }

    return roots;
}

VectorXd eigen_vector(double lambda, int n) {
    VectorXd vec(n);
    
    double x = 1;
    
    for (int i = n - 1; i >= 0; i--) {
        vec[i] = x;
        x *= lambda;
    }
    
    return vec;
}

tuple<vector<double>, vector<VectorXd>> method_danilevskovo(MatrixXd A, int n) {
    cout << "------ Method Danilevskovo ------" << endl << endl;
    auto [P, B] = frobenius_matrix(A, n);
    
    VectorXd coefs = P.row(0);
    auto characteristic_equation = characteristic_equation_builder(coefs, n);
    
    auto circles = gershgorin_circles(A, n);
    
    double min_value = INFINITY;
    double max_value = -INFINITY;
    
    for (int i = 0; i < n; i++) {
        auto [left, right] = circles[i];
        
        min_value = fmin(min_value, left);
        max_value = fmax(max_value, right);
    }
    
    vector<double> values;
    vector<double> results;
    
    for (int x = min_value; x <= max_value; x += 1) {
        values.push_back(x);
        
        double result = characteristic_equation(x);
        results.push_back(result);
    }
    
    auto characteristic_derivative = characteristic_equation_derivative_builder(coefs, n);
    
    auto roots = newton_roots(characteristic_equation, characteristic_derivative, min_value, max_value);
    
    vector<VectorXd> eigen_vectors(n);
    
    for (int i = 0; i < n; i++) {
        double lambda = roots[i];
        
        VectorXd y = eigen_vector(lambda, n);
        eigen_vectors[i] = (B * y).normalized();
    }
    
    double composition = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            composition += eigen_vectors[i].dot(eigen_vectors[j]);
        }
    }
    
    cout << "Eigen Vectors Composition: " << composition << endl;
    
    double lamdba_sum = 0;
    double lamdba_composition = 1;
    
    for (int i = 0; i < n; i++) {
        lamdba_sum += roots[i];
        lamdba_composition *= roots[i];
    }
    
    cout << "| Matrix Trace - Eigen Values Sum | = " << fabs(lamdba_sum - A.trace()) << endl;
    cout << "| Matrix Determinant - Eigen Values Composition | = " << fabs(lamdba_composition - A.determinant()) << endl;
    
    cout << endl;
    for (int i = 0; i < n; i++) {
        auto [a, b] = circles[i];
        cout << "[ " << a << " , " << b << " ] -> " << roots[i] << endl;
    }
    
    for (int i = 0; i < n; i++) {
        cout << endl << eigen_vectors[i] << endl;
    }
    
    cout << endl <<"------ END ------" << endl << endl;
    return {roots, eigen_vectors};
}

tuple<vector<double>, vector<VectorXd>> method_krylova(MatrixXd A, int n) {
    cout << "------ Method Krylova ------" << endl << endl;
    vector<VectorXd> y_vectors(n + 1, VectorXd(n));
    y_vectors[0](0) = 1;
    
    for (int i = 1; i < n + 1; i++) {
        y_vectors[i] = A * y_vectors[i - 1];
    }
    
    MatrixXd Y(n, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Y(i, j) = y_vectors[n - j - 1](i);
        }
    }
    
    VectorXd p = gauss_combined(Y, y_vectors[n], n);//Y.colPivHouseholderQr().solve(y_vectors[n]);
    
    auto characteristic_equation = characteristic_equation_builder(p, n);
    auto characteristic_derivative = characteristic_equation_derivative_builder(p, n);
    
    auto circles = gershgorin_circles(A, n);
    
    double min_value = INFINITY;
    double max_value = -INFINITY;
    
    for (int i = 0; i < n; i++) {
        auto [left, right] = circles[i];
        
        min_value = fmin(min_value, left);
        max_value = fmax(max_value, right);
    }
    
    auto lambda = newton_roots(characteristic_equation, characteristic_derivative, min_value, max_value);
    
    vector<VectorXd> eigen_vectors(n);
    
    for (int i = 0; i < n; i++) {
        double l = lambda[i];
        
        VectorXd q(n);
        q(0) = 1;
        
        for (int j = 1; j < n; j++) {
            q(j) = l * q(j - 1) - p(j - 1);
        }
        
        eigen_vectors[i] = (Y * q).normalized();
    }
    
    double composition = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            composition += eigen_vectors[i].dot(eigen_vectors[j]);
        }
    }
    
    cout << "Eigen Vectors Composition: " << composition << endl;
    
    double lamdba_sum = 0;
    double lamdba_composition = 1;
    
    for (int i = 0; i < n; i++) {
        lamdba_sum += lambda[i];
        lamdba_composition *= lambda[i];
    }
    
    cout << "| Matrix Trace - Eigen Values Sum | = " << fabs(lamdba_sum - A.trace()) << endl;
    cout << "| Matrix Determinant - Eigen Values Composition | = " << fabs(lamdba_composition - A.determinant()) << endl;
    
    cout << endl;
    for (int i = 0; i < n; i++) {
        auto [a, b] = circles[i];
        cout << "[ " << a << " , " << b << " ] -> " << lambda[i] << endl;
    }
    
    for (int i = 0; i < n; i++) {
        cout << endl << eigen_vectors[i] << endl;
    }
    
    cout << endl << "------ END ------" << endl << endl;
    return {lambda, eigen_vectors};
}

int main() {
    int n = 4;
    
    MatrixXd A = random_symmetric_matrix(n, 1000);
//    A << 2.2, 1, 0.5, 2,
//         1, 1.3, 2, 1,
//         0.5, 2, 0.5, 1.6,
//         2, 1, 1.6, 2;
    
    auto [lambda1, eigen_vectors1] = method_danilevskovo(A, n);
    auto [lambda2, eigen_vectors2] = method_krylova(A, n);
    
    double lambda_diff = 0;
    double eigen_vectors_diff = 0;
    
    for (int i = 0; i < n; i++) {
        lambda_diff += fabs(lambda1[i] - lambda2[i]);
        eigen_vectors_diff += (eigen_vectors1[i] - eigen_vectors2[i]).norm();
    }
    
    cout << "Eigen Values Diff : " << lambda_diff << endl;
    cout << "Eigen Vectors Diff : " << eigen_vectors_diff << endl;
}
