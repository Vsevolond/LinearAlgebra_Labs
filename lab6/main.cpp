#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <sciplot/sciplot.hpp>
#include <chrono>
#include <Accelerate/Accelerate.h>

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

vector<double> row_major_vector(vector<vector<double>> matrix, int n) {
    vector<double> result(n * n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = matrix[i][j];
        }
    }
    
    return result;
}

vector<double> col_major_vector(vector<vector<double>> matrix, int n) {
    vector<double> result(n * n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[j * n + i] = matrix[i][j];
        }
    }
    
    return result;
}

vector<vector<double>> row_major_matrix(vector<double> vec, int n) {
    vector<vector<double>> matrix(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = vec[i * n + j];
        }
    }
    
    return matrix;
}

vector<vector<double>> col_major_matrix(vector<double> vec, int n) {
    vector<vector<double>> matrix(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = vec[j * n + i];
        }
    }
    
    return matrix;
}

vector<double> eigen_values(vector<vector<double>> A, int n) {
    auto A_vector = row_major_vector(A, n);
    
    vector<double> values(n);
    
    char jobz = 'N';
    char uplo = 'U';
    int lwork = -1;
    
    double work_query;
    int info;
    
    dsyev_(&jobz, &uplo, &n, A_vector.data(), &n, values.data(), &work_query, &lwork, &info);
    
    lwork = work_query;
    vector<double> work(lwork);
    
    dsyev_(&jobz, &uplo, &n, A_vector.data(), &n, values.data(), work.data(), &lwork, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    return values;
}

double determinant(vector<vector<double>> A) {
    int n = (int)A.size();
    
    vector<double> A_v = col_major_vector(A, n);
    vector<double> values(n);
    
    char jobz = 'N';
    char uplo = 'U';
    int lwork = -1;
    
    double work_query;
    int info;
    
    dsyev_(&jobz, &uplo, &n, A_v.data(), &n, values.data(),
           &work_query, &lwork, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    lwork = (int)work_query;
    vector<double> work(lwork);
    
    dsyev_(&jobz, &uplo, &n, A_v.data(), &n, values.data(),
           work.data(), &lwork, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    double det = 1;
    
    for (double lambda : values) {
        det *= lambda;
    }
    
    return det;
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

// MARK: - Single Parameter Method

tuple<vector<double>, int> single_parameter_method(
    double tau,
    vector<vector<double>> A,
    vector<double> f,
    int n,
    double eps = 1e-12
) {
    vector<double> x_last(n);
    vector<double> x_curr = multiply(f, tau);
    
    int iterations = 1;
    
    while (absolute_error(x_curr, x_last) >= eps) {
        x_last = x_curr;
        x_curr = substract(
            x_last,
            multiply(
                substract(
                    multiply(A, x_last),
                    f
                ),
                tau
            )
        );
        
        iterations++;
    }
    
    return {x_curr, iterations};
}

vector<double> lapack_single_parameter_method(vector<double> A, vector<double> f, int n) {
    vector<double> A_vector = A;
    vector<double> result = f;
    
    vector<int> ipiv(n);
    int info;
    
    int nrhs = 1;
    
    dgesv_(&n, &nrhs, A_vector.data(), &n, ipiv.data(), result.data(), &n, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    return result;
}

tuple<vector<double>, int> cblas_single_parameter_method(
    double tau,
    vector<double> A,
    vector<double> f,
    int n,
    double eps = 1e-12
) {
    vector<double> x_last(n);
    vector<double> x_curr(n);
    
    cblas_dcopy(n, f.data(), 1, x_curr.data(), 1);
    cblas_dscal(n, tau, x_curr.data(), 1);
    
    int iterations = 1;
    
    vector<double> A_mul_x(n);
    vector<double> A_mul_x_sub_f(n);
    vector<double> A_mul_x_sub_f_mul_tau(n);
    
    vector<double> diff(n);
    
    cblas_dcopy(n, x_curr.data(), 1, diff.data(), 1);
    cblas_daxpy(n, -1.0, x_last.data(), 1, diff.data(), 1);
    
    double error = cblas_dnrm2(n, diff.data(), 1);
    
    while (error >= eps) {
        cblas_dcopy(n, x_curr.data(), 1, x_last.data(), 1);
        
        // A * x_last
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    n, n, 1.0, A.data(), n,
                    x_last.data(), 1, 0.0, A_mul_x.data(), 1);
        
        // A * x_last - f
        cblas_dcopy(n, A_mul_x.data(), 1, A_mul_x_sub_f.data(), 1);
        cblas_daxpy(n, -1.0, f.data(), 1, A_mul_x_sub_f.data(), 1);
        
        // tau * (A * x_last - f)
        cblas_dcopy(n, A_mul_x_sub_f.data(), 1, A_mul_x_sub_f_mul_tau.data(), 1);
        cblas_dscal(n, tau, A_mul_x_sub_f_mul_tau.data(), 1);
        
        // x_curr = x_last - tau * (A * x_last - f)
        cblas_dcopy(n, x_last.data(), 1, x_curr.data(), 1);
        cblas_daxpy(n, -1.0, A_mul_x_sub_f_mul_tau.data(), 1, x_curr.data(), 1);
        
        cblas_dcopy(n, x_curr.data(), 1, diff.data(), 1);
        cblas_daxpy(n, -1.0, x_last.data(), 1, diff.data(), 1);
        
        error = cblas_dnrm2(n, diff.data(), 1);
        
        iterations++;
    }
    
    return {x_curr, iterations};
}

tuple<vector<double>, int> zeidel_method(
    vector<vector<double>> A,
    vector<double> f,
    int n,
    double eps = 1e-12
) {
    vector<vector<double>> alpha(n, vector<double>(n));
    vector<double> betta(n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                alpha[i][j] = -A[i][j] / A[i][i];
            }
        }
        
        betta[i] = f[i] / A[i][i];
    }
    
    vector<double> x_last(n);
    vector<double> x_curr = betta;
    
    int iterations = 1;
    
    while (absolute_error(x_curr, x_last) >= eps) {
        x_last = x_curr;
        
        for (int i = 0; i < n; i++) {
            double sum = betta[i];
            
            for (int j = 0; j < n; j++) {
                if (j < i) {
                    sum += alpha[i][j] * x_curr[j];
                    
                } else if (j > i) {
                    sum += alpha[i][j] * x_last[j];
                }
            }
            
            x_curr[i] = sum;
        }
        
        iterations += 1;
    }
    
    return {x_curr, iterations};
}

// MARK: - Method Holetskovo

tuple<vector<double>, vector<vector<double>>>
method_holetskovo(vector<vector<double>> A, vector<double> f, int n) {
    for (int i = 0; i < n; i++) {
        double s = A[i][i];
        
        for (int k = 0; k < i; k++) {
            s = s - A[i][k] * A[i][k];
        }
        
        A[i][i] = sqrt(s);
        
        for (int j = i + 1; j < n; j++) {
            s = A[j][i];
            
            for (int k = 0; k < i; k++) {
                s = s - A[i][k] * A[j][k];
            }
            
            A[j][i] = s / A[i][i];
        }
    }
    
    vector<double> y(n);
    
    for (int i = 0; i < n; i++) {
        double sum = 0;
        
        for (int j = 0; j < i; j++) {
            sum += A[i][j] * y[j];
        }
        
        y[i] = (f[i] - sum) / A[i][i];
    }
    
    vector<double> x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A[j][i] * x[j];
        }
        
        x[i] = (y[i] - sum) / A[i][i];
    }
    
    return {x, A};
}

tuple<vector<double>, vector<double>>
lapack_method_holetskovo(vector<double> A, vector<double> f, int n) {
    vector<double> x = f;
    
    char uplo = 'L';
    int nrhs = 1;
    
    int info;
    
    dpotrf_(&uplo, &n, A.data(), &n, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    dpotrs_(&uplo, &n, &nrhs, A.data(), &n, x.data(), &n, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    return {x, A};
}

vector<double> cblas_method_holetskovo(vector<double> A, vector<double> f, int n) {
    for (int i = 0; i < n; i++) {
        double s = A[i * n + i];
        
        for (int k = 0; k < i; k++) {
            s = s - A[i * n + k] * A[i * n + k];
        }
        
        A[i * n + i] = sqrt(s);
        
        for (int j = i + 1; j < n; j++) {
            s = A[j * n + i];
            
            for (int k = 0; k < i; k++) {
                s = s - A[i * n + k] * A[j * n + k];
            }
            
            A[j * n + i] = s / A[i * n + i];
        }
    }
    
    vector<double> y(n);
    
    cblas_dcopy(n, f.data(), 1, y.data(), 1);
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                n, A.data(), n, y.data(), 1);
    
    vector<double> x(n);
    
    cblas_dcopy(n, y.data(), 1, x.data(), 1);
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit,
                n, A.data(), n, x.data(), 1);
    
    return x;
}

// MARK: - SVD

tuple<vector<double>, vector<double>, vector<double>>
lapack_svd(vector<double> A, int n) {
    char jobu = 'A';
    char jobvt = 'A';
    
    int info;
    int lwork = -1;
    
    vector<double> S(n);
    vector<double> U(n * n);
    vector<double> V_T(n * n);
    
    double work_query;
    dgesvd_(&jobu, &jobvt, &n, &n, A.data(), &n,
            S.data(), U.data(), &n, V_T.data(), &n,
            &work_query, &lwork, &info);
    
    lwork = (int)work_query;
    vector<double> work(lwork);
    
    dgesvd_(&jobu, &jobvt, &n, &n, A.data(), &n,
            S.data(), U.data(), &n, V_T.data(), &n,
            work.data(), &lwork, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    return {U, S, V_T};
}

// MARK: - Eigen Values & Vectors

tuple<vector<double>, vector<double>>
lapack_eigen(vector<double> A, int n) {
    char jobz = 'V';
    char uplo = 'U';
    
    int lwork = -1;
    
    double work_query;
    int info;
    
    vector<double> values(n);
    
    dsyev_(&jobz, &uplo, &n, A.data(), &n, values.data(),
           &work_query, &lwork, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    lwork = (int)work_query;
    vector<double> work(lwork);
    
    dsyev_(&jobz, &uplo, &n, A.data(), &n, values.data(),
           work.data(), &lwork, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    return {values, A};
}

// MARK: - Gauss

vector<double> gauss_combined(vector<vector<double>> A, vector<double> f, int n) {
    vector<int> permutations(n);
    for (int i = 0; i < n; i++) { permutations[i] = i; }
    
    for (int k = 0; k < n - 1; k++) {
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
        
        swap_row(A, k, row_max);
        swap_col(A, k, col_max);
        
        swap(f, k, row_max);
        swap(permutations, k, col_max);
        
        for (int row = k + 1; row < n; row++) {
            double divider = A[row][k];
            
            for (int col = k; col < n; col++) {
                A[row][col] = A[row][col] / divider - A[k][col] / A[k][k];
            }
            
            f[row] = f[row] / divider - f[k] / A[k][k];
        }
    }
    
    vector<double> x(n);
    
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        
        x[i] = (f[i] - sum) / A[i][i];
    }
    
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        int index = permutations[i];
        result[index] = x[i];
    }
    
    return result;
}

vector<double> cblas_gauss_combined(vector<double> A, vector<double> f, int n) {
    vector<int> permutations(n);
    for (int i = 0; i < n; i++) { permutations[i] = i; }
    
    for (int k = 0; k < n - 1; k++) {
        int row_max = k;
        int col_max = k;
        
        for (int row = k; row < n; row++) {
            for (int col = k; col < n; col++) {
                double curr_elem = fabs(A[row * n + col]);
                double max_elem = fabs(A[row_max * n + col_max]);
                
                if (curr_elem > max_elem) {
                    row_max = row;
                    col_max = col;
                }
            }
        }
        
        if (k != row_max) {
            cblas_dswap(n, &A[k * n], 1, &A[row_max * n], 1);
            swap(f[k], f[row_max]);
        }
        
        if (k != col_max) {
            for (int i = 0; i < n; i++) {
                swap(A[i * n + k], A[i * n + col_max]);
            }
            
            swap(permutations[k], permutations[col_max]);
        }
        
        for (int row = k + 1; row < n; row++) {
            double multiplier = A[row * n + k] / A[k * n + k];
            
            cblas_daxpy(n - k - 1, -multiplier,
                        &A[k * n + k + 1], 1,
                        &A[row * n + k + 1], 1);
            
            f[row] -= multiplier * f[k];
        }
    }
    
    vector<double> x(n);
    x[n - 1] = f[n - 1] / A[(n - 1) * n + n - 1];
    
    for (int i = n - 2; i >= 0; i--) {
        double sum = cblas_ddot(n - i - 1, &A[i * n + i + 1], 1, &x[i + 1], 1);
        x[i] = (f[i] - sum) / A[i * n + i];
    }
    
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        int index = permutations[i];
        result[index] = x[i];
    }
    
    return result;
}

vector<double> lapack_gauss_combined(vector<double> A, vector<double> f, int n) {
    vector<int> ipiv(n);
    int info;
    
    int nrhs = 1;
    
    dgesv_(&n, &nrhs, A.data(), &n, ipiv.data(), f.data(), &n, &info);
    
    if (info != 0) {
        throw runtime_error("LAPACK failed");
    }
    
    return f;
}

// MARK: - Tests

void test_single_parameter_method() {
    vector<int> dimensions;
    
    vector<double> single_durations;
    vector<double> zeidel_durations;
    vector<double> cblas_durations;
    vector<double> lapack_durations;
    
    vector<double> single_errors;
    vector<double> zeidel_errors;
    vector<double> cblas_errors;
    vector<double> lapack_errors;
    
    vector<double> single_iterations;
    vector<double> zeidel_iterations;
    vector<double> cblas_iterations;
    
    for (int n = 5; n <= 100; n += 5) {
        auto A = random_symmetric_positive_matrix(n, 10);
        auto x_correct = random_vector(n, 10);
        auto f = multiply(A, x_correct);
        
        auto lambdas = eigen_values(A, n);
        
        double max_lambda = -INFINITY;
        double min_lambda = INFINITY;
        
        for (auto lambda : lambdas) {
            max_lambda = fmax(max_lambda, fabs(lambda));
            min_lambda = fmin(min_lambda, fabs(lambda));
        }
        
        double tau = 2 / (min_lambda + max_lambda);
        
        auto [single_result, single_duration] = measure([tau, A, f, n]() {
            return single_parameter_method(tau, A, f, n);
        });
        auto [single_x, single_iteration] = single_result;
        
        auto [zeidel_result, zeidel_duration] = measure([A, f, n]() {
            return zeidel_method(A, f, n);
        });
        auto [zeidel_x, zeidel_iteration] = zeidel_result;
        
        vector<double> A_vector = row_major_vector(A, n);
        
        auto [lapack_x, lapack_duration] = measure([A_vector, f, n]() {
            return lapack_single_parameter_method(A_vector, f, n);
        });
        
        auto [cblas_result, cblas_duration] = measure([tau, A_vector, f, n]() {
            return cblas_single_parameter_method(tau, A_vector, f, n);
        });
        auto [cblas_x, cblas_iteration] = cblas_result;
        
        dimensions.push_back(n);
        
        single_durations.push_back(single_duration);
        zeidel_durations.push_back(zeidel_duration);
        lapack_durations.push_back(lapack_duration);
        cblas_durations.push_back(cblas_duration);
        
        single_iterations.push_back(single_iteration);
        zeidel_iterations.push_back(zeidel_iteration);
        cblas_iterations.push_back(cblas_iteration);
        
        double single_error = absolute_error(single_x, x_correct);
        single_errors.push_back(single_error);
        
        double zeidel_error = absolute_error(zeidel_x, x_correct);
        zeidel_errors.push_back(zeidel_error);
        
        double lapack_error = absolute_error(lapack_x, x_correct);
        lapack_errors.push_back(lapack_error);
        
        double cblas_error = absolute_error(cblas_x, x_correct);
        cblas_errors.push_back(cblas_error);
    }
    
    Plot2D plot1, plot2, plot3;

    plot1.xlabel("Matrix Dimension");
    plot1.ylabel("Absolute Error");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();

    plot1.drawCurve(dimensions, single_errors).label("Single Parameter Method");
    plot1.drawCurve(dimensions, zeidel_errors).label("Zeidel Method");
    plot1.drawCurve(dimensions, cblas_errors).label("BLAS Method");
    plot1.drawCurve(dimensions, lapack_errors).label("LAPACK Method");
    
    plot2.xlabel("Matrix Dimension");
    plot2.ylabel("Iterations");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();
    
    plot2.drawCurve(dimensions, single_iterations).label("Single Parameter Method");
    plot2.drawCurve(dimensions, zeidel_iterations).label("Zeidel Method");
    plot2.drawCurve(dimensions, cblas_iterations).label("BLAS Method");

    plot3.xlabel("Matrix Dimension");
    plot3.ylabel("Duration");
    plot3.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot3.grid().show();

    plot3.drawCurve(dimensions, single_durations).label("Single Parameter Method");
    plot3.drawCurve(dimensions, zeidel_durations).label("Zeidel Method");
    plot3.drawCurve(dimensions, cblas_durations).label("BLAS Method");
    plot3.drawCurve(dimensions, lapack_durations).label("LAPACK Method");

    Figure fig = {{plot1}, {plot2}, {plot3}};
    Canvas canvas = {{fig}};
    canvas.size(800, 1200);
    canvas.show();
}

void test_method_holetskovo() {
    vector<int> dimensions;
    
    vector<double> custom_durations;
    vector<double> lapack_durations;
    vector<double> cblas_durations;
    
    vector<double> custom_errors;
    vector<double> lapack_errors;
    vector<double> cblas_errors;
    
    vector<double> custom_decomposition_errors;
    vector<double> lapack_decomposition_errors;
    
    for (int n = 5; n <= 100; n += 5) {
        auto L = random_lower_triangle_positive_matrix(n, 1000);
        auto L_T = transposed(L);
        
        auto A = multiply(L, L_T);
        auto x_correct = random_vector(n, 1000);
        auto f = multiply(A, x_correct);
        
        auto [custom_result, custom_duration] = measure([A, f, n]() {
            return method_holetskovo(A, f, n);
        });
        auto [custom_x, custom_L] = custom_result;
        
        vector<double> A_v = row_major_vector(A, n);
        
        auto [lapack_result, lapack_duration] = measure([A_v, f, n]() {
            return lapack_method_holetskovo(A_v, f, n);
        });
        auto [lapack_x, lapack_L_v] = lapack_result;
        auto lapack_L = transposed(row_major_matrix(lapack_L_v, n));
        
        auto [cblas_x, cblas_duration] = measure([A_v, f, n]() {
            return cblas_method_holetskovo(A_v, f, n);
        });
        
        dimensions.push_back(n);
        
        custom_durations.push_back(custom_duration);
        lapack_durations.push_back(lapack_duration);
        cblas_durations.push_back(cblas_duration);
        
        double custom_error = absolute_error(custom_x, x_correct);
        custom_errors.push_back(custom_error);
        
        double lapack_error = absolute_error(lapack_x, x_correct);
        lapack_errors.push_back(lapack_error);
        
        double cblas_error = absolute_error(cblas_x, x_correct);
        cblas_errors.push_back(cblas_error);
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                custom_L[i][j] = 0;
                lapack_L[i][j] = 0;
            }
        }
        
        double custom_decomposition_error = absolute_error(custom_L, L);
        custom_decomposition_errors.push_back(custom_decomposition_error);
        
        double lapack_decomposition_error = absolute_error(lapack_L, L);
        lapack_decomposition_errors.push_back(lapack_decomposition_error);
    }
    
    Plot2D plot1, plot2, plot3;

    plot1.xlabel("Matrix Dimension");
    plot1.ylabel("Absolute Error");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();

    plot1.drawCurve(dimensions, custom_errors).label("Method Holetskovo");
    plot1.drawCurve(dimensions, lapack_errors).label("LAPACK Method");
    plot1.drawCurve(dimensions, cblas_errors).label("BLAS Method");

    plot2.xlabel("Matrix Dimension");
    plot2.ylabel("Duration");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();

    plot2.drawCurve(dimensions, custom_durations).label("Method Holetskovo");
    plot2.drawCurve(dimensions, lapack_durations).label("LAPACK Method");
    plot2.drawCurve(dimensions, cblas_durations).label("BLAS Method");
    
    plot3.xlabel("Matrix Dimension");
    plot3.ylabel("Absolute Decomposition Error");
    plot3.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot3.grid().show();
    
    plot3.drawCurve(dimensions, custom_decomposition_errors).label("Method Holetskovo");
    plot3.drawCurve(dimensions, lapack_decomposition_errors).label("LAPACK Method");

    Figure fig = {{plot1}, {plot2}, {plot3}};
    Canvas canvas = {{fig}};
    canvas.size(800, 1200);
    canvas.show();
}

void test_svd() {
    vector<int> dimensions;
    
    vector<double> A_errors;
    vector<double> durations;
    
    for (int n = 5; n <= 100; n += 5) {
        auto U = random_orthogonal_matrix(n, 1000);
        auto S = random_diagonal_matrix(n);
        auto V_T = transposed(random_orthogonal_matrix(n, 1000));
        
        auto A = multiply(U, multiply(S, V_T));
        auto A_v = col_major_vector(A, n);
        
        auto [lapack_result, duration] = measure([A_v, n]() {
            return lapack_svd(A_v, n);
        });
        auto [lapack_U_v, lapack_S_v, lapack_V_T_v] = lapack_result;
        
        vector<vector<double>> lapack_U(n, vector<double>(n, 0));
        vector<vector<double>> lapack_S(n, vector<double>(n, 0));
        vector<vector<double>> lapack_V_T(n, vector<double>(n, 0));
        
        for (int i = 0; i < n; i++) {
            lapack_S[i][i] = lapack_S_v[i];
            
            for (int j = 0; j < n; j++) {
                lapack_U[i][j] = lapack_U_v[i + j * n];
                lapack_V_T[i][j] = lapack_V_T_v[i + j * n];
            }
        }
        
        dimensions.push_back(n);
        durations.push_back(duration);
        
        auto lapack_A = multiply(lapack_U, multiply(lapack_S, lapack_V_T));
        double A_error = absolute_error(lapack_A, A);
        A_errors.push_back(A_error);
    }
    
    Plot2D plot1, plot2;

    plot1.xlabel("Matrix Dimension");
    plot1.ylabel("Absolute Error (Matrix A = U * S * V_T)");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();

    plot1.drawCurve(dimensions, A_errors).label("LAPACK Method");

    plot2.xlabel("Matrix Dimension");
    plot2.ylabel("Duration");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();

    plot2.drawCurve(dimensions, durations).label("LAPACK Method");

    Figure fig = {{plot1}, {plot2}};
    Canvas canvas = {{fig}};
    canvas.size(800, 800);
    canvas.show();
}

void test_eigen_values_and_vectors() {
    vector<double> dimensions;
    
    vector<double> lapack_trace_errors;
    vector<double> lapack_det_errors;
    vector<double> lapack_compositions;
    vector<double> lapack_durations;
    
    for (int n = 5; n <= 100; n += 5) {
        auto A = random_symmetric_matrix(n, 1000);
        auto A_v = col_major_vector(A, n);
        
        auto [lapack_result, lapack_duration] = measure([A_v, n]() {
            return lapack_eigen(A_v, n);
        });
        auto [lapack_values, lapack_vectors_v] = lapack_result;
        
        auto lapack_vectors_t = col_major_matrix(lapack_vectors_v, n);
        auto lapack_vectors = transposed(lapack_vectors_t);
        
        double trace_A = trace(A);
        double determinant_A = determinant(A);
        
        double lapack_trace_error = fabs(trace_A - sum(lapack_values));
        double lapack_det_error = fabs(determinant_A - composition(lapack_values)) / fabs(determinant_A);
        double lapack_composition = composition(lapack_vectors);
        
        dimensions.push_back(n);
        
        lapack_trace_errors.push_back(lapack_trace_error);
        lapack_det_errors.push_back(lapack_det_error);
        lapack_compositions.push_back(lapack_composition);
        lapack_durations.push_back(lapack_duration);
    }
    
    Plot2D plot1, plot2, plot3, plot4;

    plot1.xlabel("Matrix Dimension");
    plot1.ylabel("Matrix Trace - Eigen Values Sum");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();
    
    plot1.drawCurve(dimensions, lapack_trace_errors).label("LAPACK Method");

    plot2.xlabel("Matrix Dimension");
    plot2.ylabel("Matrix Determinant - Eigen Values Composition");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();

    plot2.drawCurve(dimensions, lapack_det_errors).label("LAPACK Method");
    
    plot3.xlabel("Matrix Dimension");
    plot3.ylabel("Eigen Vectors Composition");
    plot3.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot3.grid().show();

    plot3.drawCurve(dimensions, lapack_compositions).label("LAPACK Method");
    
    plot4.xlabel("Matrix Dimension");
    plot4.ylabel("Duration");
    plot4.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot4.grid().show();

    plot4.drawCurve(dimensions, lapack_durations).label("LAPACK Method");

    Figure fig = {{plot1}, {plot2}, {plot3}, {plot4}};
    Canvas canvas = {{fig}};
    canvas.size(800, 1600);
    canvas.show();
}

void test_gauss() {
    vector<double> dimensions;
    
    vector<double> gauss_errors;
    vector<double> cblas_errors;
    vector<double> lapack_errors;
    
    vector<double> gauss_durations;
    vector<double> cblas_durations;
    vector<double> lapack_durations;
    
    for (int n = 5; n <= 100; n += 5) {
        auto A = random_matrix(n, 1000);
        auto x_true = random_vector(n, 1000);
        auto f = multiply(A, x_true);
        
        auto [gauss_x, gauss_duration] = measure([A, f, n]() {
            return gauss_combined(A, f, n);
        });
        
        auto A_row = row_major_vector(A, n);
        auto [cblas_x, cblas_duration] = measure([A_row, f, n]() {
            return cblas_gauss_combined(A_row, f, n);
        });
        
        auto A_col = col_major_vector(A, n);
        auto [lapack_x, lapack_duration] = measure([A_col, f, n]() {
            return lapack_gauss_combined(A_col, f, n);
        });
        
        auto gauss_error = absolute_error(gauss_x, x_true);
        auto cblas_error = absolute_error(cblas_x, x_true);
        auto lapack_error = absolute_error(lapack_x, x_true);
        
        dimensions.push_back(n);
        
        gauss_errors.push_back(gauss_error);
        cblas_errors.push_back(cblas_error);
        lapack_errors.push_back(lapack_error);
        
        gauss_durations.push_back(gauss_duration);
        cblas_durations.push_back(cblas_duration);
        lapack_durations.push_back(lapack_duration);
    }
    
    Plot2D plot1, plot2;

    plot1.xlabel("Matrix Dimension");
    plot1.ylabel("Absolute Error");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();

    plot1.drawCurve(dimensions, gauss_errors).label("Gauss Combined");
    plot1.drawCurve(dimensions, cblas_errors).label("BLAS Method");
    plot1.drawCurve(dimensions, lapack_errors).label("LAPACK Method");

    plot2.xlabel("Matrix Dimension");
    plot2.ylabel("Duration");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();

    plot2.drawCurve(dimensions, gauss_durations).label("Gauss Combined");
    plot2.drawCurve(dimensions, cblas_durations).label("BLAS Method");
    plot2.drawCurve(dimensions, lapack_durations).label("LAPACK Method");

    Figure fig = {{plot1}, {plot2}};
    Canvas canvas = {{fig}};
    canvas.size(800, 800);
    canvas.show();
}

void test_gauss_dominance() {
    vector<double> dominances;
    
    vector<double> gauss_errors;
    vector<double> cblas_errors;
    vector<double> lapack_errors;
    
    vector<double> gauss_durations;
    vector<double> cblas_durations;
    vector<double> lapack_durations;
    
    int n = 100;
    auto x_true = random_vector(n, 1000);
    
    for (double alpha = -100; alpha <= 100; alpha += 10) {
        auto A = random_matrix(n, alpha, 1000);
        auto f = multiply(A, x_true);
        
        auto [gauss_x, gauss_duration] = measure([A, f, n]() {
            return gauss_combined(A, f, n);
        });
        
        auto A_row = row_major_vector(A, n);
        auto [cblas_x, cblas_duration] = measure([A_row, f, n]() {
            return cblas_gauss_combined(A_row, f, n);
        });
        
        auto A_col = col_major_vector(A, n);
        auto [lapack_x, lapack_duration] = measure([A_col, f, n]() {
            return lapack_gauss_combined(A_col, f, n);
        });
        
        auto gauss_error = absolute_error(gauss_x, x_true);
        auto cblas_error = absolute_error(cblas_x, x_true);
        auto lapack_error = absolute_error(lapack_x, x_true);
        
        dominances.push_back(alpha);
        
        gauss_errors.push_back(gauss_error);
        cblas_errors.push_back(cblas_error);
        lapack_errors.push_back(lapack_error);
        
        gauss_durations.push_back(gauss_duration);
        cblas_durations.push_back(cblas_duration);
        lapack_durations.push_back(lapack_duration);
    }
    
    Plot2D plot1, plot2;

    plot1.xlabel("Matrix Dominance");
    plot1.ylabel("Absolute Error");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();

    plot1.drawCurve(dominances, gauss_errors).label("Gauss Combined");
    plot1.drawCurve(dominances, cblas_errors).label("BLAS Method");
    plot1.drawCurve(dominances, lapack_errors).label("LAPACK Method");

    plot2.xlabel("Matrix Dominance");
    plot2.ylabel("Duration");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();

    plot2.drawCurve(dominances, gauss_durations).label("Gauss Combined");
    plot2.drawCurve(dominances, cblas_durations).label("BLAS Method");
    plot2.drawCurve(dominances, lapack_durations).label("LAPACK Method");

    Figure fig = {{plot1}, {plot2}};
    Canvas canvas = {{fig}};
    canvas.size(800, 800);
    canvas.show();
}

int main() {
    test_single_parameter_method();
    test_method_holetskovo();
    test_svd();
    test_eigen_values_and_vectors();
    test_gauss();
    test_gauss_dominance();
}
