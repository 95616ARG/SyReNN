/*********************                                                        */
/*! \file main.cpp
** \verbatim
** Top contributors (to current version):
**   Guy Katz
** This file is part of the Reluplex project.
** Copyright (c) 2016-2017 by the authors listed in the file AUTHORS
** (in the top-level source directory) and their institutional affiliations.
** All rights reserved. See the file COPYING in the top-level source
** directory for licensing information.\endverbatim
**/

#include <assert.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <signal.h>

#include "Engine.h"
#include "FloatUtils.h"
#include "InputQuery.h"
#include "ReluConstraint.h"
#include "TimeUtils.h"
#include "ReluplexError.h"

using VecMatrix = std::vector<std::vector<double>>;
VecMatrix readMatrix(std::ifstream *file) {
    int rows = -1, cols = -1;
    (*file) >> rows >> cols;
    VecMatrix mat(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*file) >> mat[i][j];
        }
    }
    return mat;
}

struct Index
{
    Index(unsigned layer, unsigned node)
        : layer(layer), node(node)
    {
    }

    unsigned layer;
    unsigned node;

    bool operator<( const Index &other ) const
    {
        if (layer != other.layer)
            return layer < other.layer;
        if (node != other.node)
            return node < other.node;
        return false;
    }
};

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: "
                  << argv[0] << " <net.nnet> <spec.nspec> <steps> <out_file>"
                  << std::endl;
        return 1;
    }

    char *netPath = argv[1];
    char *specPath = argv[2];
    unsigned steps = std::atoi(argv[3]);

    std::ofstream out_file(argv[4]);
    out_file << netPath << "," << specPath << "," << steps << "," << std::flush;

    std::ifstream net(netPath);
    std::vector<char> layer_types;
    std::vector<VecMatrix> layer_weights;
    std::vector<std::vector<double>> layer_biases;

    unsigned weighted_nodes = 0,
             relu_pairs = 0;

    while (layer_types.empty() || layer_types.back() != 'T') {
        char layer_type = '\0';
        while (!(layer_type == 'A' || layer_type == 'T' || layer_type == 'R')) {
            net >> layer_type;
        }
        layer_types.push_back(layer_type);
        switch (layer_type) {
            case 'A':
                layer_weights.push_back(readMatrix(&net));
                layer_biases.push_back(readMatrix(&net).front());
                weighted_nodes += layer_biases.back().size();
                break;
            case 'T':
                layer_weights.push_back(readMatrix(&net));
                layer_weights.push_back(readMatrix(&net));
                weighted_nodes += layer_weights.back().size();
                break;
            case 'R':
                relu_pairs += layer_biases.back().size();
                break;
        }
    }

    std::ifstream spec(specPath);
    VecMatrix inputConstraints = readMatrix(&spec);
    VecMatrix safeConstraints = readMatrix(&spec);
    VecMatrix outputConstraints = readMatrix(&spec);
    spec.close();

    const unsigned net_inputs = 2;

    unsigned tableau_size =
        // 1. Input vars appear once.
        net_inputs +
        // 2. Each weighted node has a var *PER STEP*.
        (steps * weighted_nodes) +
        // 3. Each pre-ReLU var has a post-ReLU pair var.
        (steps * relu_pairs) +
        // 4. One aux var per input constraint.
        inputConstraints.size() +
        // 5. One aux var per output safe constraint *PER STEP*.
        ((steps - 1) * safeConstraints.size()) +
        // 6. One aux var per output constraint *AT THE LAST STEP*.
        outputConstraints.size();
    // NOTE: Used to have a single extra for the constants.

    InputQuery inputQuery;
    inputQuery.setNumberOfVariables(tableau_size);

    unsigned running_index = 0;

    Map<Index, unsigned> nodeToVars;

    // First, add the input vars.
    for (unsigned i = 0; i < net_inputs; i++) {
        nodeToVars[Index(0, i)] = running_index++;
        // NOTE: This *ASSUMES* these are valid bounds on the input state!
        inputQuery.setLowerBound(nodeToVars[Index(0, i)], -100.0);
        inputQuery.setUpperBound(nodeToVars[Index(0, i)], +100.0);
    }

    unsigned layer = 1;
    std::vector<int> transition_layers;

    for (unsigned step = 0; step < steps; step++) {
    unsigned input_layer = layer - 1;
    unsigned w_i = 0, b_i = 0;
    for (char layer_type : layer_types) {
        if (layer_type == 'A') {
            auto &weights = layer_weights[w_i++];
            auto &biases = layer_biases[b_i++];
            // Weights is (n_inputs x n_outputs), biases is (n_outputs)
            for (unsigned output = 0; output < biases.size(); output++) {
                nodeToVars[Index(layer, output)] = running_index++;
                unsigned var = nodeToVars[Index(layer, output)];
                inputQuery.setLowerBound(var, FloatUtils::negativeInfinity());
                inputQuery.setUpperBound(var, FloatUtils::infinity());

                Equation node_equation;
                node_equation.addAddend(-1., var);
                for (unsigned input = 0; input < weights.size(); input++) {
                    unsigned from_var = nodeToVars[Index(layer - 1, input)];
                    node_equation.addAddend(weights[input][output], from_var);
                }
                node_equation.setScalar(-biases[output]);
                inputQuery.addEquation(node_equation);
            }
        } else if (layer_type == 'T') {
            auto &transition_A = layer_weights[w_i++];
            auto &transition_B = layer_weights[w_i++];
            // transition_A is (out_state x in_state), transition_B is (state x action)
            for (unsigned output = 0; output < transition_A.size(); output++) {
                nodeToVars[Index(layer, output)] = running_index++;
                unsigned var = nodeToVars[Index(layer, output)];
                inputQuery.setLowerBound(var, FloatUtils::negativeInfinity());
                inputQuery.setUpperBound(var, FloatUtils::infinity());

                Equation node_equation;
                node_equation.addAddend(-1., var);

                for (unsigned state_input = 0; state_input < transition_A.front().size(); state_input++) {
                    unsigned from_var = nodeToVars[Index(input_layer, state_input)];
                    if (state_input == output) {
                        node_equation.addAddend(
                            1.0 + transition_A[output][state_input], from_var);
                    } else {
                        node_equation.addAddend(
                            transition_A[output][state_input], from_var);
                    }
                }
                for (unsigned action_input = 0; action_input < transition_B.front().size(); action_input++) {
                    unsigned from_var = nodeToVars[Index(layer - 1, action_input)];
                    node_equation.addAddend(
                        transition_B[output][action_input], from_var);
                }
                node_equation.setScalar(0.0);
                inputQuery.addEquation(node_equation);
            }
            transition_layers.push_back(layer);
        } else if (layer_type == 'R') {
            unsigned last_nodes = layer_biases[w_i - 1].size();
            for (unsigned output = 0; output < last_nodes; output++) {
                nodeToVars[Index(layer, output)] = running_index++;
                unsigned preNode = nodeToVars[Index(layer - 1, output)];
                unsigned postNode = nodeToVars[Index(layer, output)];
                ReluConstraint *relu_pair = new ReluConstraint(preNode, postNode);
                inputQuery.addPiecewiseLinearConstraint(relu_pair);
                inputQuery.setLowerBound(postNode, 0.0);
                inputQuery.setUpperBound(postNode, FloatUtils::infinity());
            }
        }
        layer++;
    }
    }

    // Set constraints for inputs.
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
        unsigned constraint_index = running_index++;
        inputQuery.setUpperBound(constraint_index, 0.0);
        Equation input_equation;
        input_equation.addAddend(-1., constraint_index);
        for (unsigned int j = 0; j < inputConstraints[i].size() - 1; j++) {
            input_equation.addAddend(
                inputConstraints[i][j], nodeToVars[Index(0, j)]);
        }
        input_equation.setScalar(-inputConstraints[i].back());
        inputQuery.addEquation(input_equation);
    }

    for (unsigned l = 0; l < transition_layers.size(); l++) {
        auto &layer = transition_layers[l];
        if (l < (transition_layers.size() - 1)) {
            // Set constraints for non-final outputs.
            for (unsigned i = 0; i < safeConstraints.size(); i++) {
                unsigned constraint_index = running_index++;
                inputQuery.setUpperBound(constraint_index, 0.0);
                Equation output_equation;
                output_equation.addAddend(-1., constraint_index);
                for (unsigned int j = 0; j < safeConstraints[i].size() - 1; j++) {
                    output_equation.addAddend(
                        safeConstraints[i][j], nodeToVars[Index(layer, j)]);
                }
                output_equation.setScalar(-safeConstraints[i].back());
                inputQuery.addEquation(output_equation);
            }
        } else {
            // Set constraints for final outputs.
            for (unsigned i = 0; i < outputConstraints.size(); i++) {
                unsigned constraint_index = running_index++;
                inputQuery.setUpperBound(constraint_index, 0.0);
                Equation output_equation;
                output_equation.addAddend(-1., constraint_index);
                for (unsigned int j = 0; j < outputConstraints[i].size() - 1; j++) {
                    output_equation.addAddend(
                            outputConstraints[i][j],
                            nodeToVars[Index(layer, j)]);
                }
                output_equation.setScalar(-outputConstraints[i].back());
                inputQuery.addEquation(output_equation);
            }
        }
    }

    timespec start = TimeUtils::sampleMicro();
    timespec end;

    try {
        Engine engine;
        if (engine.processInputQuery(inputQuery) && engine.solve()) {
            engine.extractSolution(inputQuery);

            out_file << "SAT";
            printf("Solution found!\n\n");
        }
        else {
            out_file << "UNS";
            printf("Can't solve!\n");
        }
    } catch (ReluplexError &e) {
        printf("Error: %d\n", e.getCode());
    }

    end = TimeUtils::sampleMicro();

    unsigned microPassed = TimeUtils::timePassed(start, end);
    unsigned seconds = microPassed / 1000000;
    unsigned minutes = seconds / 60;
    unsigned hours = minutes / 60;
    out_file << "," << (((double)microPassed) / 1000000) << std::endl;
    out_file.close();

    printf("Total run time: %llu micro (%02u:%02u:%02u)\n",
           TimeUtils::timePassed(start, end),
           hours,
           minutes - (hours * 60),
           seconds - (minutes * 60));

    return 0;
}
