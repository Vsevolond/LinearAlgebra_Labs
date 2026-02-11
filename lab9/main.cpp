#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <chrono>
#include <sciplot/sciplot.hpp>

using namespace std;
using sciplot::Plot2D;
using sciplot::Figure;
using sciplot::Canvas;

class Matrix {
public:
    size_t n;
    vector<double> data;
    
    Matrix(size_t size, double value = 0.0) : n(size), data(size * size, value) {}
    
    double& operator()(size_t i, size_t j) { return data[i * n + j]; }
    const double& operator()(size_t i, size_t j) const { return data[i * n + j]; }
};

Matrix random_matrix(int n, double limit) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-limit, limit);
    
    Matrix result(n);
    for (auto& val : result.data) {
        val = dist(gen);
    }
    return result;
}

Matrix multiply(const Matrix &A, const Matrix &B) {
    size_t n = A.n;
    Matrix C(n);
    
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double a_ik = A(i, k);
            for (size_t j = 0; j < n; j++) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
    return C;
}

Matrix vinograd(const Matrix &A, const Matrix &B) {
    size_t n = A.n;
    size_t m = n / 2;
    Matrix C(n);
    
    vector<double> row_factor(n, 0.0);
    vector<double> col_factor(n, 0.0);
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            row_factor[i] += A(i, 2*j) * A(i, 2*j+1);
            col_factor[i] += B(2*j, i) * B(2*j+1, i);
        }
    }
    
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double sum = -row_factor[i] - col_factor[k];
            for (size_t j = 0; j < m; j++) {
                sum += (A(i, 2*j) + B(2*j+1, k)) * (A(i, 2*j+1) + B(2*j, k));
            }
            C(i, k) = sum;
        }
    }
    
    if (n % 2 != 0) {
        size_t last = n - 1;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                C(i, j) += A(i, last) * B(last, j);
            }
        }
    }
    return C;
}

void add_block(const Matrix &src, Matrix &dst,
               size_t src_row, size_t src_col,
               size_t dst_row, size_t dst_col, size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            dst(dst_row + i, dst_col + j) += src(src_row + i, src_col + j);
        }
    }
}

void sub_block(const Matrix &src1, const Matrix &src2, Matrix &dst,
               size_t row1, size_t col1,
               size_t row2, size_t col2,
               size_t dst_row, size_t dst_col, size_t size) {
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            dst(dst_row + i, dst_col + j) = src1(row1 + i, col1 + j) - src2(row2 + i, col2 + j);
        }
    }
}

void strassen_impl(const Matrix &A, const Matrix &B, Matrix &C, size_t n, size_t n_min) {
    if (n <= n_min) {
        for (size_t i = 0; i < n; i++) {
            for (size_t k = 0; k < n; k++) {
                double a_val = A(i, k);
                for (size_t j = 0; j < n; j++) {
                    C(i, j) += a_val * B(k, j);
                }
            }
        }
        return;
    }
    
    size_t m = n / 2;
    Matrix P1(m), P2(m), P3(m), P4(m), P5(m), P6(m), P7(m);
    Matrix tempA(m), tempB(m);
    
    // P1 = (A11 + A22) * (B11 + B22)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            tempA(i, j) = A(i, j) + A(m + i, m + j);
            tempB(i, j) = B(i, j) + B(m + i, m + j);
        }
    }
    strassen_impl(tempA, tempB, P1, m, n_min);
    
    // P2 = (A21 + A22) * B11
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            tempA(i, j) = A(m + i, j) + A(m + i, m + j);
            tempB(i, j) = B(i, j);
        }
    }
    strassen_impl(tempA, tempB, P2, m, n_min);
    
    // P3 = A11 * (B12 - B22)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            tempA(i, j) = A(i, j);
            tempB(i, j) = B(i, m + j) - B(m + i, m + j);
        }
    }
    strassen_impl(tempA, tempB, P3, m, n_min);
    
    // P4 = A22 * (B21 - B11)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            tempA(i, j) = A(m + i, m + j);
            tempB(i, j) = B(m + i, j) - B(i, j);
        }
    }
    strassen_impl(tempA, tempB, P4, m, n_min);
    
    // P5 = (A11 + A12) * B22
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            tempA(i, j) = A(i, j) + A(i, m + j);
            tempB(i, j) = B(m + i, m + j);
        }
    }
    strassen_impl(tempA, tempB, P5, m, n_min);
    
    // P6 = (A21 - A11) * (B11 + B12)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            tempA(i, j) = A(m + i, j) - A(i, j);
            tempB(i, j) = B(i, j) + B(i, m + j);
        }
    }
    strassen_impl(tempA, tempB, P6, m, n_min);
    
    // P7 = (A12 - A22) * (B21 + B22)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            tempA(i, j) = A(i, m + j) - A(m + i, m + j);
            tempB(i, j) = B(m + i, j) + B(m + i, m + j);
        }
    }
    strassen_impl(tempA, tempB, P7, m, n_min);
    
    // C11 = P1 + P4 - P5 + P7
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            C(i, j) = P1(i, j) + P4(i, j) - P5(i, j) + P7(i, j);
        }
    }
    
    // C12 = P3 + P5
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            C(i, m + j) = P3(i, j) + P5(i, j);
        }
    }
    
    // C21 = P2 + P4
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            C(m + i, j) = P2(i, j) + P4(i, j);
        }
    }
    
    // C22 = P1 - P2 + P3 + P6
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            C(m + i, m + j) = P1(i, j) - P2(i, j) + P3(i, j) + P6(i, j);
        }
    }
}

Matrix strassen(const Matrix &A, const Matrix &B, size_t n_min) {
    size_t n = A.n;
    Matrix C(n);
    strassen_impl(A, B, C, n, n_min);
    return C;
}

template<typename Func>
auto measure(Func&& func) {
    auto start = chrono::steady_clock::now();
    auto result = func();
    auto end = chrono::steady_clock::now();
    return make_pair(std::move(result), chrono::duration<double, milli>(end - start).count());
}

int main() {
    const vector<int> minimums = {2, 32, 64, 128, 256, 1024};
    const vector<int> dimensions = {2, 4, 8, 16, 32, 64, 128, 512, 1024};
    
    vector<Matrix> A_matrices, B_matrices;
    for (int n : dimensions) {
        A_matrices.push_back(random_matrix(n, 1000));
        B_matrices.push_back(random_matrix(n, 1000));
    }
    
    vector<vector<double>> base_durations(minimums.size(), vector<double>(dimensions.size()));
    vector<vector<double>> vinograd_durations(minimums.size(), vector<double>(dimensions.size()));
    vector<vector<double>> strass_durations(minimums.size(), vector<double>(dimensions.size()));
    
    for (size_t min_idx = 0; min_idx < minimums.size(); ++min_idx) {
        int n_min = minimums[min_idx];
        cout << "Testing n_min = " << n_min << endl;
        
        for (size_t dim_idx = 0; dim_idx < dimensions.size(); ++dim_idx) {
            int n = dimensions[dim_idx];
            cout << "  Matrix size: " << n << "x" << n << endl;
            
            const auto& A = A_matrices[dim_idx];
            const auto& B = B_matrices[dim_idx];
            
            auto [C_base, base_time] = measure([&]() { return multiply(A, B); });
            base_durations[min_idx][dim_idx] = base_time;
            
            auto [C_vinograd, vinograd_time] = measure([&]() { return vinograd(A, B); });
            vinograd_durations[min_idx][dim_idx] = vinograd_time;
            
            auto [C_strass, strass_time] = measure([&]() { return strassen(A, B, n_min); });
            strass_durations[min_idx][dim_idx] = strass_time;
        }
    }
    
    Plot2D plot1, plot2, plot3, plot4, plot5, plot6, plot7;
    
    plot1.xlabel("Matrix Dimension");
    plot1.ylabel("Duration (n_{min} = 2)");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();
    
    plot1.drawCurve(dimensions, base_durations[0]).label("Basic");
    plot1.drawCurve(dimensions, strass_durations[0]).label("Strassen");
    plot1.drawCurve(dimensions, vinograd_durations[0]).label("Vinograd");
    
    plot2.xlabel("Matrix Dimension");
    plot2.ylabel("Duration (n_{min} = 32)");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();
    
    plot2.drawCurve(dimensions, base_durations[1]).label("Basic");
    plot2.drawCurve(dimensions, strass_durations[1]).label("Strassen");
    plot2.drawCurve(dimensions, vinograd_durations[1]).label("Vinograd");
    
    plot3.xlabel("Matrix Dimension");
    plot3.ylabel("Duration (n_{min} = 64)");
    plot3.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot3.grid().show();
    
    plot3.drawCurve(dimensions, base_durations[2]).label("Basic");
    plot3.drawCurve(dimensions, strass_durations[2]).label("Strassen");
    plot3.drawCurve(dimensions, vinograd_durations[2]).label("Vinograd");
    
    plot4.xlabel("Matrix Dimension");
    plot4.ylabel("Duration (n_{min} = 128)");
    plot4.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot4.grid().show();
    
    plot4.drawCurve(dimensions, base_durations[3]).label("Basic");
    plot4.drawCurve(dimensions, strass_durations[3]).label("Strassen");
    plot4.drawCurve(dimensions, vinograd_durations[3]).label("Vinograd");
    
    plot5.xlabel("Matrix Dimension");
    plot5.ylabel("Duration (n_{min} = 256)");
    plot5.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot5.grid().show();
    
    plot5.drawCurve(dimensions, base_durations[4]).label("Basic");
    plot5.drawCurve(dimensions, strass_durations[4]).label("Strassen");
    plot5.drawCurve(dimensions, vinograd_durations[4]).label("Vinograd");
    
    plot6.xlabel("Matrix Dimension");
    plot6.ylabel("Duration (n_{min} = 1024)");
    plot6.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot6.grid().show();
    
    plot6.drawCurve(dimensions, base_durations[5]).label("Basic");
    plot6.drawCurve(dimensions, strass_durations[5]).label("Strassen");
    plot6.drawCurve(dimensions, vinograd_durations[5]).label("Vinograd");
    
    plot7.xlabel("N_min");
    plot7.ylabel("Delta (Classic - Strassen)");
    plot7.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot7.grid().show();
    
    vector<double> diff_durations(minimums.size());
    
    for (size_t i = 0; i < minimums.size(); ++i) {
        double sum = 0;
        for (size_t j = 3; j < dimensions.size(); ++j) {
            sum += base_durations[i][j] - strass_durations[i][j];
        }
        diff_durations[i] = sum / (dimensions.size() - 3);
    }
    
    plot7.drawCurve(minimums, diff_durations);

    Figure fig = {{plot1}, {plot2}, {plot3}, {plot4}, {plot5}, {plot6}, {plot7}};
    Canvas canvas = {{fig}};
    canvas.size(800, 2800);
    canvas.show();
}
