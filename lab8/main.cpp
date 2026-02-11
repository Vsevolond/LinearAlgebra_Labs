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
using std::uniform_int_distribution;
using std::random_device;
using std::swap;
using std::fmin;
using std::fmax;
using std::min;
using std::max;
using std::fabs;
using std::abs;
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

template<typename Func>
auto measure(Func func) -> tuple<decltype(func()), double> {
    auto start = steady_clock::now();
    auto result = func();
    auto end = steady_clock::now();
    
    duration<double, micro> time = end - start;
    return {result, time.count()};
}

int diff(vector<int> a, vector<int> b) {
    int n = max((int)a.size(), (int)b.size());
    
    int diff = 0;
    
    for (int i = 0; i < n; i++) {
        diff += abs(a[i] - b[i]);
    }
    
    return diff;
}

vector<int> random_number(int n) {
    random_device rd;
    mt19937 gen(rd());
    
    uniform_int_distribution<int> dist_null(0, 9);
    uniform_int_distribution<int> dist(1, 9);
    
    vector<int> number(n, 0);
    number[0] = dist_null(gen);
    
    for (int i = 1; i < n; i++) {
        number[i] = dist(gen);
    }
    
    return number;
}

vector<int> base_multiply(vector<int> a, vector<int> b) {
    vector<int> res(a.size() + b.size(), 0);
    
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b.size(); j++) {
            res[i + j] += a[i] * b[j];
        }
    }
    
    int carry = 0;
    
    for (int i = 0; i < res.size(); i++) {
        int t = res[i] + carry;
        
        res[i] = t % 10;
        carry = t / 10;
    }
    
    while (res.size() > 1 && res.back() == 0) {
        res.pop_back();
    }
    
    return res;
}

vector<int> add(vector<int> a, vector<int> b) {
    vector<int> result(max(a.size(), b.size()) + 1, 0);
    
    int carry = 0;
    
    for (int i = 0; i < result.size() - 1; i++) {
        int sum = carry;
        
        if (i < a.size()) sum += a[i];
        if (i < b.size()) sum += b[i];
        
        result[i] = sum % 10;
        carry = sum / 10;
    }
    
    if (carry) { result.back() = carry; }
    
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }
    
    return result;
}

vector<int> subtract(vector<int> a, vector<int> b) {
    vector<int> result = a;
    int borrow = 0;
    
    for (int i = 0; i < a.size(); ++i) {
        int diff = a[i] - borrow;
        
        if (i < b.size()) { diff -= b[i]; }
        
        if (diff < 0) {
            diff += 10;
            borrow = 1;
            
        } else {
            borrow = 0;
        }
        
        result[i] = diff;
    }
    
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }
    
    return result;
}

vector<int> shift_left(vector<int> a, int k) {
    vector<int> result(k + a.size(), 0);
    
    copy(a.begin(), a.end(), result.begin() + k);
    
    return result;
}

vector<int> method_karazubi(vector<int> a, vector<int> b) {
    int a_size = (int)a.size();
    int b_size = (int)b.size();
    
    int n = (int)max(a.size(), b.size());
    
    if (n <= 64) {
        return base_multiply(a, b);
    }
    
    int m = (n + 1) / 2;
    
    /// a = a1 * 10^m + a0
    vector<int> a0(a.begin(), a.begin() + min(m, a_size));
    vector<int> a1(a.size() > m ? a.begin() + m : a.end(), a.end());
    
    /// b = b1 * 10^m + b0
    vector<int> b0(b.begin(), b.begin() + min(m, b_size));
    vector<int> b1(b.size() > m ? b.begin() + m : b.end(), b.end());
    
    if (a1.empty()) { a1 = {0}; }
    if (b1.empty()) { b1 = {0}; }
    
    /// p0 = a0 * b0
    auto p0 = method_karazubi(a0, b0);
    ///p2 = a1 * b1
    auto p2 = method_karazubi(a1, b1);
    
    auto sum_a = add(a0, a1);
    auto sum_b = add(b0, b1);
    
    /// p1 = (a0 + a1) * (b0 + b1)
    auto p1 = method_karazubi(sum_a, sum_b);
    
    /// mid = p1 - p0 - p2
    auto mid = subtract(subtract(p1, p0), p2);
    
    /// p2 * 10^2m
    auto part2 = shift_left(p2, 2 * m);
    /// mid * 10^m
    auto part1 = shift_left(mid, m);
    /// p0
    auto part0 = p0;
    
    /// result = p2 * 10^2m + mid * 10^m + p0
    auto result = add(part2, part1);
    result = add(result, part0);
    
    return result;
}

int main() {
    vector<int> dimensions;
    
    vector<int> base_durations;
    vector<int> karazuba_durations;
    
    vector<int> diffs;
    
    for (int n = 2; n <= 8192; n *= 2) {
        auto a = random_number(n);
        auto b = random_number(n);
        
        auto [base_result, base_duration] = measure([a, b]() {
            return base_multiply(a, b);
        });
        
        auto [karazuba_result, karazuba_duration] = measure([a, b]() {
            return method_karazubi(a, b);
        });
        
        int delta = diff(base_result, karazuba_result);
        
        dimensions.push_back(n);
        
        base_durations.push_back(base_duration);
        karazuba_durations.push_back(karazuba_duration);
        
        diffs.push_back(delta);
    }
    
    Plot2D plot1, plot2;

    plot1.xlabel("Dimension");
    plot1.ylabel("Duration");
    plot1.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot1.grid().show();

    plot1.drawCurve(dimensions, base_durations).label("Base Multiply");
    plot1.drawCurve(dimensions, karazuba_durations).label("Method Karazubi");
    
    plot2.xlabel("Dimension");
    plot2.ylabel("Diff");
    plot2.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);
    plot2.grid().show();
    
    plot2.drawCurve(dimensions, diffs);

    Figure fig = {{plot1}, {plot2}};
    Canvas canvas = {{fig}};
    canvas.size(800, 800);
    canvas.show();
}
