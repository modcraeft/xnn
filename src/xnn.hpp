// src/xnn.hpp — Modern C++ header-only neural network (zero warnings, RAII, STL)
#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>

enum class Activation { Sigmoid, Tanh, ReLU, Softmax };
enum class Loss { MSE, CrossEntropy };

struct Matrix {
    std::vector<float> data;
    size_t rows = 0, cols = 0;

    Matrix() = default;
    Matrix(size_t r, size_t c) : data(r * c), rows(r), cols(c) {}

    float& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    const float& operator()(size_t i, size_t j) const { return data[i * cols + j]; }
    float* row_ptr(size_t i) { return data.data() + i * cols; }
    void fill(float v) { std::fill(data.begin(), data.end(), v); }
    void randn(float scale = 1.0f) {
        static std::mt19937 gen{std::random_device{}()};
        std::normal_distribution<float> dist(0, scale);
        for (auto& x : data) x = dist(gen);
    }
};

class Network {
public:
    std::vector<size_t> arch;
    std::vector<Activation> activations;
    Loss loss_type = Loss::MSE;

    std::vector<Matrix> W, b, a;  // weights, biases, activations

    Network(const std::vector<size_t>& architecture,
            const std::vector<Activation>& acts,
            Loss loss = Loss::MSE)
        : arch(architecture), activations(acts), loss_type(loss)
    {
        if (arch.size() < 2) return;

        a.resize(arch.size());
        a[0] = Matrix(arch[0], 1);

        W.reserve(arch.size() - 1);
        b.reserve(arch.size() - 1);
        for (size_t i = 1; i < arch.size(); ++i) {
            size_t fan_in = arch[i-1], fan_out = arch[i];
            float limit = (activations[i] == Activation::ReLU)
                ? std::sqrt(2.0f / fan_in)
                : std::sqrt(6.0f / (fan_in + fan_out));

            W.emplace_back(fan_out, fan_in);
            b.emplace_back(fan_out, 1);

            W.back().randn(limit);
            b.back().randn(0.1f);
            a[i] = Matrix(fan_out, 1);
        }
    }

    void forward(const float* input = nullptr) {
        if (input) std::memcpy(a[0].data.data(), input, arch[0] * sizeof(float));

        for (size_t i = 1; i < arch.size(); ++i) {
            // z = W * a_prev + b
            for (size_t j = 0; j < arch[i]; ++j) {
                float sum = b[i-1](j, 0);
                for (size_t k = 0; k < arch[i-1]; ++k)
                    sum += W[i-1](j, k) * a[i-1](k, 0);
                a[i](j, 0) = sum;
            }

            // activation
            switch (activations[i]) {
                case Activation::Sigmoid:
                    for (size_t j = 0; j < arch[i]; ++j)
                        a[i](j, 0) = 1.0f / (1.0f + std::exp(-a[i](j, 0)));
                    break;
                case Activation::Tanh:
                    for (size_t j = 0; j < arch[i]; ++j)
                        a[i](j, 0) = std::tanh(a[i](j, 0));
                    break;
                case Activation::ReLU:
                    for (size_t j = 0; j < arch[i]; ++j)
                        if (a[i](j, 0) < 0) a[i](j, 0) = 0;
                    break;
                case Activation::Softmax: {
                    float max_val = a[i](0, 0);
                    for (size_t j = 1; j < arch[i]; ++j)
                        max_val = std::max(max_val, a[i](j, 0));
                    float sum = 0;
                    for (size_t j = 0; j < arch[i]; ++j) {
                        a[i](j, 0) = std::exp(a[i](j, 0) - max_val);
                        sum += a[i](j, 0);
                    }
                    for (size_t j = 0; j < arch[i]; ++j)
                        a[i](j, 0) /= sum;
                    break;
                }
            }
        }
    }

    void predict(const float* input, float* output) {
        forward(input);
        std::memcpy(output, a.back().data.data(), arch.back() * sizeof(float));
    }
};
