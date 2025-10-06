#include <iostream>
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
using std::tuple;
using std::numeric_limits;
using std::mt19937;
using std::uniform_real_distribution;
using std::random_device;
using std::swap;
using std::max;
using std::fabs;
using std::sqrt;
using std::srand;
using std::iter_swap;
using std::cout;
using std::endl;

// MARK: - Support Methods

double absolute_error(vector<double> dst, vector<double> src, int n) {
    double error = 0;
    
    for (int i = 0; i < n; i++) {
        double diff = dst[i] - src[i];
        error += diff * diff;
    }
    
    return sqrt(error);
}

double absolute_error(vector<double> dst, VectorXd src, int n) {
    double error = 0;
    
    for (int i = 0; i < n; i++) {
        double diff = dst[i] - src(i);
        error += diff * diff;
    }
    
    return sqrt(error);
}

double absolute_error(VectorXd dst, VectorXd src, int n) {
    double error = 0;
    
    for (int i = 0; i < n; i++) {
        double diff = dst(i) - src(i);
        error += diff * diff;
    }
    
    return sqrt(error);
}

double euclide_norm(vector<double> v, int n) {
    double norm = 0;
    
    for (int i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    
    return sqrt(norm);
}

double euclide_norm(VectorXd v, int n) {
    double norm = 0;
    
    for (int i = 0; i < n; i++) {
        norm += v(i) * v(i);
    }
    
    return sqrt(norm);
}

double relative_error(vector<double> dst, vector<double> src, int n) {
    double absolute_err = absolute_error(dst, src, n);
    double src_norm = euclide_norm(src, n);
    
    return absolute_err / src_norm;
}

double relative_error(vector<double> dst, VectorXd src, int n) {
    double absolute_err = absolute_error(dst, src, n);
    double src_norm = euclide_norm(src, n);
    
    return absolute_err / src_norm;
}

double relative_error(VectorXd dst, VectorXd src, int n) {
    double absolute_err = absolute_error(dst, src, n);
    double src_norm = euclide_norm(src, n);
    
    return absolute_err / src_norm;
}

template<typename T>
void swap(vector<T> &v, int i, int j) {
    if (i != j) {
        iter_swap(v.begin() + i, v.begin() + j);
    }
}

template<typename T>
void swap_row(vector<vector<T>> &A, int i, int j) {
    if (i != j) {
        iter_swap(A.begin() + i, A.begin() + j);
    }
}

template<typename T>
void swap_col(vector<vector<T>> &A, int i, int j) {
    if (i != j) {
        int n = A.size();
        for (int row = 0; row < n; row++) { swap(A[row], i, j); }
    }
}

template<typename T>
vector<vector<T>> convert_matrix(MatrixXd m, int n) {
    vector<vector<T>> converted(n, vector<T>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            converted[i][j] = m(i, j);
        }
    }
    
    return converted;
}

template<typename T>
vector<T> convert_vector(VectorXd v, int n) {
    vector<T> converted(n);
    
    for (int i = 0; i < n; i++) {
        converted[i] = v(i);
    }
    
    return converted;
}

// MARK: - Классический метод Гаусса

tuple<vector<double>, double> gauss_classic(vector<vector<double>> A, vector<double> f, int n) {
    
    vector<vector<double>> A_ref = A;
    
    // MARK: - Поиск максимального элемента матрицы в начале
    
    double max_0 = A[0][0];
    
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            double elem = fabs(A[row][col]);
            
            if (elem > max_0) { max_0 = elem; }
        }
    }
    
    // MARK: - Прямой ход
    
    double max_k = A[0][0];
    
    for (int k = 0; k < n - 1; k++) {
        
        // MARK: - Исключения
        
        for (int row = k + 1; row < n; row++) {
            double divider = A[row][k];
            
            for (int col = k; col < n; col++) {
                A[row][col] = A[row][col] / divider - A[k][col] / A[k][k];
            }
            
            f[row] = f[row] / divider - f[k] / A[k][k];
        }
        
        // MARK: - Поиск максимального элемента матрицы на k-ом шаге
        
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                double elem = fabs(A[row][col]);
                
                if (elem > max_k) { max_k = elem; }
            }
        }
    }
    
    double g_coef = max_k / max_0;
    
    // MARK: - Обратный ход
    
    vector<double> x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        
        x[i] = (f[i] - sum) / A[i][i];
    }
    
    return {x, g_coef};
}

// MARK: - Метод Гаусса с выбором главного элемента по строке

tuple<vector<double>, double> gauss_row(vector<vector<double>> A, vector<double> f, int n) {
    
    vector<vector<double>> A_ref = A;
    
    // MARK: - Поиск максимального элемента матрицы в начале
    
    double max_0 = A[0][0];
    
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            double elem = fabs(A[row][col]);
            
            if (elem > max_0) { max_0 = elem; }
        }
    }
    
    // MARK: - Прямой ход
    
    vector<int> permutations(n);
    for (int i = 0; i < n; i++) { permutations[i] = i; }
    
    double max_k = A[0][0];
    
    for (int k = 0; k < n - 1; k++) {
        
        // MARK: - Поиск главного элемента
        
        int col_max = k;
        
        for (int col = k + 1; col < n; col++) {
            double col_elem = fabs(A[k][col]);
            double col_max_elem = fabs(A[k][col_max]);
            
            if (col_elem > col_max_elem) { col_max = col; }
        }
        
        // MARK: - Перестановка столбцов
        
        swap_col(A, k, col_max);
        swap(permutations, k, col_max);
        
        // MARK: - Исключения
        
        for (int row = k + 1; row < n; row++) {
            double divider = A[row][k];
            
            for (int col = k; col < n; col++) {
                A[row][col] = A[row][col] / divider - A[k][col] / A[k][k];
            }
            
            f[row] = f[row] / divider - f[k] / A[k][k];
        }
        
        // MARK: - Поиск максимального элемента матрицы на k-ом шаге
        
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                double elem = fabs(A[row][col]);
                
                if (elem > max_k) { max_k = elem; }
            }
        }
    }
    
    double g_coef = max_k / max_0;
    
    // MARK: - Обратный ход
    
    vector<double> x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        
        x[i] = (f[i] - sum) / A[i][i];
    }
    
    // MARK: - Восстановление вектора
    
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        int index = permutations[i];
        result[index] = x[i];
    }
    
    return {result, g_coef};
}

// MARK: - Метод Гаусса с выбором главного элемента по столбцу

tuple<vector<double>, double> gauss_col(vector<vector<double>> A, vector<double> f, int n) {
    
    vector<vector<double>> A_ref = A;
    
    // MARK: - Поиск максимального элемента матрицы в начале
    
    double max_0 = A[0][0];
    
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            double elem = fabs(A[row][col]);
            
            if (elem > max_0) { max_0 = elem; }
        }
    }
    
    // MARK: - Прямой ход
    
    double max_k = A[0][0];
    
    for (int k = 0; k < n - 1; k++) {
        
        // MARK: - Поиск главного элемента
        
        int row_max = k;
        
        for (int row = k + 1; row < n; row++) {
            double row_elem = fabs(A[row][k]);
            double row_max_elem = fabs(A[row_max][k]);
            
            if (row_elem > row_max_elem) { row_max = row; }
        }
        
        // MARK: - Перестановка строк
        
        swap_row(A, k, row_max);
        swap(f, k, row_max);
        
        // MARK: - Исключения
        
        for (int row = k + 1; row < n; row++) {
            double divider = A[row][k];
            
            for (int col = k; col < n; col++) {
                A[row][col] = A[row][col] / divider - A[k][col] / A[k][k];
            }
            
            f[row] = f[row] / divider - f[k] / A[k][k];
        }
        
        // MARK: - Поиск максимального элемента матрицы на k-ом шаге
        
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                double elem = fabs(A[row][col]);
                
                if (elem > max_k) { max_k = elem; }
            }
        }
    }
    
    double g_coef = max_k / max_0;
    
    // MARK: - Обратный ход
    
    vector<double> x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        
        x[i] = (f[i] - sum) / A[i][i];
    }
    
    return {x, g_coef};
}

// MARK: - Метод Гаусса с выбором главного элемента по матрице

tuple<vector<double>, double> gauss_combined(vector<vector<double>> A, vector<double> f, int n) {
    
    vector<vector<double>> A_ref = A;
    
    // MARK: - Поиск максимального элемента матрицы в начале
    
    double max_0 = A[0][0];
    
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            double elem = fabs(A[row][col]);
            
            if (elem > max_0) { max_0 = elem; }
        }
    }
    
    // MARK: - Прямой ход
    
    vector<int> permutations(n);
    for (int i = 0; i < n; i++) { permutations[i] = i; }
    
    double max_k = A[0][0];
    
    for (int k = 0; k < n - 1; k++) {
        
        // MARK: - Поиск главного элемента
        
        int row_max = k;
        int col_max = k;
        
        for (int row = k; row < n; row++) {
            for (int col = k; col < n; col++) {
                double curr_elem = fabs(A[row][col]);
                double max_elem = fabs(A[row_max][col_max]);
                
                if (curr_elem > max_elem) {
                    row_max = row;
                    col_max = col;
                }
            }
        }
        
        // MARK: - Перестановка строк и столбцов
        
        swap_row(A, k, row_max);
        swap_col(A, k, col_max);
        
        swap(f, k, row_max);
        swap(permutations, k, col_max);
        
        // MARK: - Исключения
        
        for (int row = k + 1; row < n; row++) {
            double divider = A[row][k];
            
            for (int col = k; col < n; col++) {
                A[row][col] = A[row][col] / divider - A[k][col] / A[k][k];
            }
            
            f[row] = f[row] / divider - f[k] / A[k][k];
        }
        
        // MARK: - Поиск максимального элемента матрицы на k-ом шаге
        
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                double elem = fabs(A[row][col]);
                
                if (elem > max_k) { max_k = elem; }
            }
        }
    }
    
    double g_coef = max_k / max_0;
    
    // MARK: - Обратный ход
    
    vector<double> x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        
        x[i] = (f[i] - sum) / A[i][i];
    }
    
    // MARK: - Восстановление вектора
    
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        int index = permutations[i];
        result[index] = x[i];
    }
    
    return {result, g_coef};
}

// MARK: - Генерация матрицы

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

// MARK: - Test

void test(int n) {
    VectorXd x_correct = random_vector(n, 1000);
    
    MatrixXd A_ref = random_matrix(n, 1000);
    VectorXd f_ref = A_ref * x_correct;
    
    auto A = convert_matrix<double>(A_ref, n);
    auto f = convert_vector<double>(f_ref, n);

    VectorXd x_ref = A_ref.colPivHouseholderQr().solve(f_ref);
    
    auto [x_classic, coef_classic] = gauss_classic(A, f, n);
    auto [x_row, coef_row] = gauss_row(A, f, n);
    auto [x_col, coef_col] = gauss_col(A, f, n);
    auto [x_combined, coef_combined] = gauss_combined(A, f, n);
    
    double ref_absolute_error = absolute_error(x_ref, x_correct, n);
    double classic_absolute_error = absolute_error(x_classic, x_correct, n);
    double row_absolute_error = absolute_error(x_row, x_correct, n);
    double col_absolute_error = absolute_error(x_col, x_correct, n);
    double combined_absolute_error = absolute_error(x_combined, x_correct, n);
    
    cout << "Eigen Method Error: " << ref_absolute_error << endl;
    cout << "Classic Method Error: " << classic_absolute_error << endl;
    cout << "Row Pivoting Error: " << row_absolute_error << endl;
    cout << "Column Pivoting Error: " << col_absolute_error << endl;
    cout << "Combined Pivoting Error: " << combined_absolute_error << endl;
}

void test(int n, double dominance, double step) {
    vector<double> ref_absolute_errors;
    vector<double> classic_absolute_errors;
    vector<double> row_absolute_errors;
    vector<double> col_absolute_errors;
    vector<double> combined_absolute_errors;
    
    vector<double> classic_calculation_errors;
    vector<double> row_calculation_errors;
    vector<double> col_calculation_errors;
    vector<double> combined_calculation_errors;
    
    vector<double> ref_perturbation_errors;
    vector<double> classic_perturbation_errors;
    vector<double> row_perturbation_errors;
    vector<double> col_perturbation_errors;
    vector<double> combined_perturbation_errors;
    vector<double> perturbation_error_limits;
    
    vector<double> alphas;
    
    double perturbation_level = 1e-8;
    
    VectorXd x_correct = random_vector(n, 1000);
    
    for (double alpha = -dominance; alpha <= dominance; alpha += step) {
        
        cout << "Matrix Diagonal Dominance: " << alpha << endl;
        alphas.push_back(alpha);
        
        MatrixXd A_ref = random_matrix(n, alpha, 1000);
        VectorXd f_ref = A_ref * x_correct;
        
        auto A = convert_matrix<double>(A_ref, n);
        auto f = convert_vector<double>(f_ref, n);
        
        // MARK: - Решение СЛАУ
        
        VectorXd x_ref = A_ref.colPivHouseholderQr().solve(f_ref);
        
        auto [x_classic, coef_classic] = gauss_classic(A, f, n);
        auto [x_row, coef_row] = gauss_row(A, f, n);
        auto [x_col, coef_col] = gauss_col(A, f, n);
        auto [x_combined, coef_combined] = gauss_combined(A, f, n);
        
        // MARK: - Вычисление абсолютных погрешностей
        
        double ref_absolute_error = absolute_error(x_ref, x_correct, n);
        double classic_absolute_error = absolute_error(x_classic, x_correct, n);
        double row_absolute_error = absolute_error(x_row, x_correct, n);
        double col_absolute_error = absolute_error(x_col, x_correct, n);
        double combined_absolute_error = absolute_error(x_combined, x_correct, n);
        
        ref_absolute_errors.push_back(ref_absolute_error);
        classic_absolute_errors.push_back(classic_absolute_error);
        row_absolute_errors.push_back(row_absolute_error);
        col_absolute_errors.push_back(col_absolute_error);
        combined_absolute_errors.push_back(combined_absolute_error);
        
        // MARK: - Вычисление погрешностей прямых методов
        
        double conditionality_value = A_ref.norm() * A_ref.inverse().norm();
        cout << "Matrix Conditionality Value: " << conditionality_value << endl << endl;
        double factor = numeric_limits<double>::epsilon();
        
        cout << "----------- Error Rates -----------" << endl;
        cout << "Classic Gaussian:   " << coef_classic << endl;
        cout << "Row Pivoting:       " << coef_row << endl;
        cout << "Column Pivoting:    " << coef_col << endl;
        cout << "Combined Pivoting:  " << coef_combined << endl;
        cout << endl;
        
        double classic_calculation_error = (conditionality_value * n * coef_classic * factor);
        double row_calculation_error = (conditionality_value * n * coef_row * factor);
        double col_calculation_error = (conditionality_value * n * coef_col * factor);
        double combined_calculation_error = (conditionality_value * n * coef_combined * factor);
        
        classic_calculation_errors.push_back(classic_calculation_error);
        row_calculation_errors.push_back(row_calculation_error);
        col_calculation_errors.push_back(col_calculation_error);
        combined_calculation_errors.push_back(combined_calculation_error);
        
        // MARK: - Вычисление погрешностей при возмущении
        
        MatrixXd A_noise = random_matrix(n, 1000);
        VectorXd f_noise = random_vector(n, 1000);
        
        MatrixXd A_perturbed_ref = A_ref + A_noise * perturbation_level * A_ref.norm();
        VectorXd f_perturbed_ref = f_ref + f_noise * perturbation_level * f_ref.norm();
        
        double A_relative_error = (A_ref - A_perturbed_ref).norm() / A_ref.norm();
        double f_relative_error = (f_ref - f_perturbed_ref).norm() / f_ref.norm();
        
        double perturbation_error_limit = conditionality_value * (A_relative_error + f_relative_error);
        perturbation_error_limits.push_back(perturbation_error_limit);
        
        auto A_perturbed = convert_matrix<double>(A_perturbed_ref, n);
        auto f_perturbed = convert_vector<double>(f_perturbed_ref, n);
        
        VectorXd x_perturbed_ref = A_perturbed_ref.colPivHouseholderQr().solve(f_perturbed_ref);
        
        auto [x_perturbed_classic, _] = gauss_classic(A_perturbed, f_perturbed, n);
        auto [x_perturbed_row, _] = gauss_row(A_perturbed, f_perturbed, n);
        auto [x_perturbed_col, _] = gauss_col(A_perturbed, f_perturbed, n);
        auto [x_perturbed_combined, _] = gauss_combined(A_perturbed, f_perturbed, n);
        
        double ref_perturbation_error = relative_error(x_perturbed_ref, x_ref, n);
        double classic_perturbation_error = relative_error(x_perturbed_classic, x_classic, n);
        double row_perturbation_error = relative_error(x_perturbed_row, x_row, n);
        double col_perturbation_error = relative_error(x_perturbed_col, x_col, n);
        double combined_perturbation_error = relative_error(x_perturbed_combined, x_combined, n);
        
        ref_perturbation_errors.push_back(ref_perturbation_error);
        classic_perturbation_errors.push_back(classic_perturbation_error);
        row_perturbation_errors.push_back(row_perturbation_error);
        col_perturbation_errors.push_back(col_perturbation_error);
        combined_perturbation_errors.push_back(combined_perturbation_error);
    }
    
    // MARK: - Построение графиков
    
    plt::figure();
    plt::plot(alphas, classic_absolute_errors, {{"label", "Classic Gaussian"}});
    plt::plot(alphas, row_absolute_errors, {{"label", "Row Pivoting"}});
    plt::plot(alphas, col_absolute_errors, {{"label", "Column Pivoting"}});
    plt::plot(alphas, combined_absolute_errors, {{"label", "Combined Pivoting"}});
    plt::plot(alphas, ref_absolute_errors, {{"label", "Eigen Library"}});
    plt::xlabel("Alpha (Diagonal Dominance)");
    plt::ylabel("Absolute Error (Euclidean Norm)");
    plt::title("Absolute Error vs Diagonal Dominance");
    plt::legend();
    plt::grid(true);
    
    plt::figure();
    plt::plot(alphas, classic_calculation_errors, {{"label", "Classic Gaussian"}});
    plt::plot(alphas, row_calculation_errors, {{"label", "Row Pivoting"}});
    plt::plot(alphas, col_calculation_errors, {{"label", "Column Pivoting"}});
    plt::plot(alphas, combined_calculation_errors, {{"label", "Combined Pivoting"}});
    plt::xlabel("Alpha (Diagonal Dominance)");
    plt::ylabel("Calculation Error");
    plt::title("Calculation Error vs Diagonal Dominance");
    plt::legend();
    plt::grid(true);
    
    plt::figure();
    plt::plot(alphas, classic_perturbation_errors, {{"label", "Classic Gaussian"}});
    plt::plot(alphas, row_perturbation_errors, {{"label", "Row Pivoting"}});
    plt::plot(alphas, col_perturbation_errors, {{"label", "Column Pivoting"}});
    plt::plot(alphas, combined_perturbation_errors, {{"label", "Combined Pivoting"}});
    plt::plot(alphas, ref_perturbation_errors, {{"label", "Eigen Library"}});
    plt::plot(alphas, perturbation_error_limits, {{"label", "Perturbation Error Limit"}});
    plt::xlabel("Alpha (Diagonal Dominance)");
    plt::ylabel("Perturbation Error (Relative Error)");
    plt::title("Perturbation Error vs Diagonal Dominance");
    plt::legend();
    plt::grid(true);
    
    plt::show();
}

// MARK: - Main

int main() {
    test(10, 10, 0.5);
//    test(5);
}
