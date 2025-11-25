#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <Eigen/Dense>
#include <sciplot/sciplot.hpp>

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

double newton_single_root(function<double(double)> f, function<double(double)> df, double x0, double tol = 1e-6) {
    double x = x0;

    while (true) {
        double y = f(x);
        double dy = df(x);

        if (fabs(dy) < 1e-12) {
            break;
        }

        double x_new = x - y / dy;

        if (fabs(x_new - x) < tol) {
            return x_new;
        }

        x = x_new;
    }

    return NAN;
}

vector<double> newton_roots(function<double(double)> f, function<double(double)> df, double start, double end, double step = 0.1, double eps=1e-6) {
    vector<double> roots;

    for (double x0 = start; x0 <= end; x0 += step) {
        double root = newton_single_root(f, df, x0, eps);

        if (!std::isnan(root)) {
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

int main() {
    int n = 4;
    
    MatrixXd A = random_symmetric_matrix(n, 1000);
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
    
    for (int i = 0; i < n; i++) {
        auto [a, b] = circles[i];
        cout << "[ " << a << " , " << b << " ] -> " << roots[i] << endl;
    }
    
    Plot2D plot;
        
    vector<double> roots_y(n, 0.0);
    plot.drawPoints(roots, roots_y).label("Roots").pointType(7).pointSize(3);
    
    vector<std::string> colors = {"blue", "green", "orange", "purple", "brown"};
    for (int i = 0; i < n; ++i) {
        auto [left, right] = circles[i];
        vector<double> x_interval = {left, right};
        vector<double> y_interval = {0.1 + 0.1 * i, 0.1 + 0.1 * i};
        string color = colors[i % colors.size()];
        
        plot.drawCurve(x_interval, y_interval)
            .label("Interval " + std::to_string(i))
            .lineColor(color)
            .lineWidth(4);
    }
    
    plot.drawCurve(values, results).label("Characteristic Equation");
    
    plot.xlabel("x");
    plot.ylabel("y");
    plot.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot.grid().show();
    
    Figure fig = {{plot}};
    Canvas canvas = {{fig}};
    canvas.size(800, 600);
    canvas.show();
    
    return 0;
}
