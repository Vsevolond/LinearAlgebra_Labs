#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    plt::plot({1,3,2,4});
    plt::show();
}
