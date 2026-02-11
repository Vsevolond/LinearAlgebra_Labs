#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <sciplot/sciplot.hpp>
#include <chrono>

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
using std::min;
using std::max;
using std::fabs;
using std::sqrt;
using std::srand;
using std::iter_swap;
using std::cout;
using std::endl;
using std::isnan;
using std::string;
using std::micro;
using std::runtime_error;

using std::chrono::duration;
using std::chrono::steady_clock;

vector<double> multiply(vector<vector<double>> matrix, vector<double> vec) {
    auto n = min(matrix.size(), vec.size());
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        double sum = 0;
        
        for (int j = 0; j < n; j++) {
            sum += matrix[i][j] * vec[j];
        }
        
        result[i] = sum;
    }
    
    return result;
}

vector<vector<double>> multiply(vector<vector<double>> dst, vector<vector<double>> src) {
    auto n = min(dst.size(), src.size());
    vector<vector<double>> result(n, vector<double>(n, 0));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += dst[i][k] * src[k][j];
            }
        }
    }
    
    return result;
}

vector<double> sum(vector<double> dst, vector<double> src) {
    auto n = min(dst.size(), src.size());
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        result[i] = dst[i] + src[i];
    }
    
    return result;
}

double sum(vector<double> vec) {
    auto n = vec.size();
    double sum = 0;
    
    for (int i = 0; i < n; i++) {
        sum += vec[i];
    }
    
    return sum;
}

vector<double> substract(vector<double> dst, vector<double> src) {
    auto n = min(dst.size(), src.size());
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        result[i] = dst[i] - src[i];
    }
    
    return result;
}

double composition(vector<double> vec) {
    auto n = vec.size();
    double result = 1;
    
    for (int i = 0; i < n; i++) {
        result *= vec[i];
    }
    
    return result;
}

vector<vector<double>> random_symmetric_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    vector<vector<double>> matrix(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) { matrix[i][j] = matrix[j][i]; }
            else { matrix[i][j] = dist(gen); }
        }
    }
    
    return matrix;
}

vector<vector<double>> random_symmetric_positive_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    vector<vector<double>> matrix(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) { matrix[i][j] = matrix[j][i]; }
            else { matrix[i][j] = dist(gen); }
            
            if (i == j) { matrix[i][j] += n * limit; }
        }
    }
    
    return matrix;
}

vector<vector<double>> random_matrix(int n, double alpha, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    vector<vector<double>> matrix(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        double sum = 0;
        
        for (int j = 0; j < n; j++) {
            double elem = dist(gen);
            matrix[i][j] = elem;
            
            if (j != i) { sum += fabs(elem); }
        }
        
        double elem = sum + alpha;
        matrix[i][i] = (matrix[i][i] >= 0) ? elem : -elem;
    }
    
    return matrix;
}

vector<vector<double>> random_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    vector<vector<double>> matrix(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = dist(gen);
        }
    }
    
    return matrix;
}

vector<vector<double>> random_lower_triangle_positive_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    vector<vector<double>> matrix(n, vector<double>(n, 0));
    
    for (int i = 0; i < n; i++) {
        matrix[i][i] = dist(gen) + n * limit;
        
        for (int j = 0; j < i; j++) {
            matrix[i][j] = dist(gen);
        }
    }
    
    return matrix;
}

vector<vector<double>> identity_matrix(int n) {
    vector<vector<double>> result(n, vector<double>(n, 0));
    
    for (int i = 0; i < n; i++) {
        result[i][i] = 1;
    }
    
    return result;
}

vector<vector<double>> similarity_matrix(vector<vector<double>> D, int n, int col) {
    vector<vector<double>> B(n, vector<double>(n, 0));
    
    for (int j = 0; j < n; j++) {
        if (j == col) {
            B[col][j] = 1 / D[col + 1][col];
            
        } else {
            B[j][j] = 1;
            B[col][j] = -D[col + 1][j] / D[col + 1][col];
        }
    }
    
    return B;
}

vector<vector<double>> similarity_inverted_matrix(vector<vector<double>> D, int n, int col) {
    vector<vector<double>> B_inverted(n, vector<double>(n, 0));
    
    for (int j = 0; j < n; j++) {
        B_inverted[col][j] = D[col + 1][j];
        
        if (j != col) {
            B_inverted[j][j] = 1;
        }
    }
    
    return B_inverted;
}

vector<vector<double>> random_orthogonal_matrix(int n, double limit) {
    auto U = random_matrix(n, limit);
    
    for (int j = 0; j < n; j++) {
        double sum = 0;
        
        for (int i = 0; i < n; i++) {
            sum += U[i][j] * U[i][j];
        }
        
        double norm = sqrt(sum);
        
        for (int i = 0; i < n; i++) {
            U[i][j] /= norm;
        }
        
        for (int k = j + 1; k < n; k++) {
            double dot_product = 0;
            
            for (int i = 0; i < n; i++) {
                dot_product += U[i][j] * U[i][k];
            }
            
            for (int i = 0; i < n; i++) {
                U[i][k] -= dot_product * U[i][j];
            }
        }
    }
    
    return U;
}

vector<vector<double>> random_diagonal_matrix(
    int n,
    double min_singular = 0.1,
    double max_singular = 100.0
) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(min_singular, max_singular);
    
    vector<vector<double>> S(n, vector<double>(n, 0));
    
    for (int i = 0; i < n; i++) {
        double t = static_cast<double>(i) / (n - 1);
        S[i][i] = min_singular * (1.0 - t) + max_singular * t;
    }
    
    return S;
}

vector<double> random_vector(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    vector<double> vector(n);
    
    for (int i = 0; i < n; i++) {
        vector[i] = dist(gen);
    }
    
    return vector;
}

vector<double> multiply(vector<double> vec, double factor) {
    auto n = vec.size();
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        result[i] = vec[i] * factor;
    }
    
    return result;
}

double multiply(vector<double> src, vector<double> dst) {
    auto n = min(src.size(), dst.size());
    double result = 0;
    
    for (int i = 0; i < n; i++) {
        result += src[i] * dst[i];
    }
    
    return result;
}

double absolute_error(vector<double> dst, vector<double> src) {
    auto n = min(dst.size(), src.size());
    double error = 0;
    
    for (int i = 0; i < n; i++) {
        double diff = dst[i] - src[i];
        error += diff * diff;
    }
    
    return sqrt(error);
}

double absolute_error(vector<vector<double>> dst, vector<vector<double>> src) {
    auto n = min(dst.size(), src.size());
    double error = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = dst[i][j] - src[i][j];
            error += diff * diff;
        }
    }
    
    return sqrt(error);
}

void transpose(vector<vector<double>> &matrix) {
    auto n = matrix.size();
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            swap(matrix[i][j], matrix[j][i]);
        }
    }
}

vector<vector<double>> transposed(vector<vector<double>> matrix) {
    auto n = matrix.size();
    
    vector<vector<double>> result(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j< n; j++) {
            result[i][j] = matrix[j][i];
        }
    }
    
    return result;
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

template<typename Func>
auto measure(Func func) -> tuple<decltype(func()), double> {
    auto start = steady_clock::now();
    auto result = func();
    auto end = steady_clock::now();
    
    duration<double, micro> time = end - start;
    return {result, time.count()};
}

double trace(vector<vector<double>> A) {
    auto n = A.size();
    double result = 0;
    
    for (int i = 0; i < n; i++) {
        result += A[i][i];
    }
    
    return result;
}

double composition(vector<vector<double>> vectors) {
    auto n = vectors.size();
    double result = 0;
    
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            result += multiply(vectors[i], vectors[j]);
        }
    }
    
    return result;
}

vector<vector<double>> random_lower_triangle_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    vector<vector<double>> matrix(n, vector<double>(n, 0));
    
    for (int i = 0; i < n; i++) {
        matrix[i][i] = 1;
        
        for (int j = 0; j < i; j++) {
            matrix[i][j] = dist(gen);
        }
    }
    
    return matrix;
}

vector<vector<double>> random_upper_triangle_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    vector<vector<double>> matrix(n, vector<double>(n, 0));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            matrix[i][j] = dist(gen);
        }
    }
    
    return matrix;
}

tuple<vector<vector<double>>, vector<vector<double>>>
lu_decomposition(vector<vector<double>> A, int n) {
    vector<vector<double>> L(n, vector<double>(n, 0));
    vector<vector<double>> U(n, vector<double>(n, 0));
    
    for (int k = 0; k < n; k++) {
        /// U[k][j]
        for (int j = k; j < n; j++) {
            double sum = 0;
            for (int l = 0; l < k; l++) {
                sum += L[k][l] * U[l][j];
            }
                
            U[k][j] = A[k][j] - sum;
        }
        
        /// L[i][k]
        L[k][k] = 1;
        
        for (int i = k + 1; i < n; i++) {
            double sum = 0;
            for (int l = 0; l < k; l++) {
                sum += L[i][l] * U[l][k];
            }
            
            L[i][k] = (A[i][k] - sum) / U[k][k];
        }
    }
    
    return {L, U};
}

int main() {
    vector<int> dimensions;
    vector<double> errors;
    
    for (int n = 5; n <= 100; n += 5) {
        auto L = random_lower_triangle_matrix(n, 1000);
        auto U = random_upper_triangle_matrix(n, 1000);
        
        auto A = multiply(L, U);
        
        auto [L_res, U_res] = lu_decomposition(A, n);
        
        auto A_res = multiply(L_res, U_res);
        
        double error = absolute_error(A, A_res);
        
        dimensions.push_back(n);
        errors.push_back(error);
    }
    
    Plot2D plot1;

    plot1.xlabel("Matrix Dimension");
    plot1.ylabel("Absolute Error");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();

    plot1.drawCurve(dimensions, errors);

    Figure fig = {{plot1}};
    Canvas canvas = {{fig}};
    canvas.size(800, 400);
    canvas.show();
}
