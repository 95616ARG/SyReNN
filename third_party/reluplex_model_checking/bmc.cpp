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

#include "AcasNeuralNetwork.h"
#include "File.h"
#include "Reluplex.h"
#include "MString.h"

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

Reluplex *lastReluplex = NULL;

void got_signal(int) {
    std::cout << "Got signal\n" << std::endl;

    if (lastReluplex) {
        lastReluplex->quit();
    }
}

int main(int argc, char **argv) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = got_signal;
    sigfillset(&sa.sa_mask);
    sigaction(SIGQUIT, &sa, NULL);

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
        // 2. Each weighted node has a var and an aux var for the equation *PER
        // STEP*.
        (steps * 2 * weighted_nodes) +
        // 3. Each pre-ReLU var has a post-ReLU pair var.
        (steps * relu_pairs) +
        // 4. One aux var per input constraint.
        inputConstraints.size() +
        // 5. One aux var per output safe constraint *PER STEP*.
        ((steps - 1) * safeConstraints.size()) +
        // 6. One aux var per output constraint *AT THE LAST STEP*.
        outputConstraints.size() +
        // 7. A single variable for the constants.
        1;

    Reluplex reluplex(tableau_size);
    lastReluplex = &reluplex;

    unsigned running_index = 0;
    // Add a constant-1 variable.
    unsigned constantVar = running_index++;
    reluplex.setLowerBound(constantVar, 1.0);
    reluplex.setUpperBound(constantVar, 1.0);

    Map<Index, unsigned> nodeToVars;
    Map<Index, unsigned> nodeToAux;

    // First, add the input vars.
    for (unsigned i = 0; i < net_inputs; i++) {
        nodeToVars[Index(0, i)] = running_index++;
        // NOTE: This *ASSUMES* these are valid bounds on the input state!
        reluplex.setLowerBound(nodeToVars[Index(0, i)], -100.0);
        reluplex.setUpperBound(nodeToVars[Index(0, i)], +100.0);
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
                nodeToAux[Index(layer, output)] = running_index++;

                unsigned auxVar = nodeToAux[Index(layer, output)];
                reluplex.initializeCell(auxVar, auxVar, -1);
                reluplex.initializeCell(auxVar, nodeToVars[Index(layer, output)], -1.0);
                for (unsigned input = 0; input < weights.size(); input++) {
                    unsigned from_var = nodeToVars[Index(layer - 1, input)];
                    reluplex.initializeCell(auxVar, from_var, weights[input][output]);
                }
                reluplex.initializeCell(auxVar, constantVar, biases[output]);

                reluplex.markBasic(auxVar);
                reluplex.setLowerBound(auxVar, 0.0);
                reluplex.setUpperBound(auxVar, 0.0);
            }
        } else if (layer_type == 'T') {
            auto &transition_A = layer_weights[w_i++];
            auto &transition_B = layer_weights[w_i++];
            // transition_A is (out_state x in_state), transition_B is (state x action)
            for (unsigned output = 0; output < transition_A.size(); output++) {
                nodeToVars[Index(layer, output)] = running_index++;
                nodeToAux[Index(layer, output)] = running_index++;

                unsigned auxVar = nodeToAux[Index(layer, output)];
                reluplex.initializeCell(auxVar, auxVar, -1);
                reluplex.initializeCell(auxVar, nodeToVars[Index(layer, output)], -1);

                for (unsigned state_input = 0; state_input < transition_A.front().size(); state_input++) {
                    unsigned from_var = nodeToVars[Index(input_layer, state_input)];
                    if (state_input == output) {
                        reluplex.initializeCell(auxVar, from_var, 1.0 + transition_A[output][state_input]);
                    } else {
                        reluplex.initializeCell(auxVar, from_var, transition_A[output][state_input]);
                    }
                }
                for (unsigned action_input = 0; action_input < transition_B.front().size(); action_input++) {
                    unsigned from_var = nodeToVars[Index(layer - 1, action_input)];
                    reluplex.initializeCell(auxVar, from_var, transition_B[output][action_input]);
                }

                reluplex.markBasic(auxVar);
                reluplex.setLowerBound(auxVar, 0.0);
                reluplex.setUpperBound(auxVar, 0.0);
            }
            transition_layers.push_back(layer);
        } else if (layer_type == 'R') {
            unsigned last_nodes = layer_biases[w_i - 1].size();
            for (unsigned output = 0; output < last_nodes; output++) {
                nodeToVars[Index(layer, output)] = running_index++;
                unsigned preNode = nodeToVars[Index(layer - 1, output)];
                unsigned postNode = nodeToVars[Index(layer, output)];
                reluplex.setReluPair(preNode, postNode);
                reluplex.setLowerBound(postNode, 0.0);
            }
        }
        layer++;
    }
    }

    // Set constraints for inputs.
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
        unsigned constraint_index = running_index++;
        reluplex.markBasic(constraint_index); // ?
        reluplex.setUpperBound(constraint_index, 0.0);
        reluplex.initializeCell(constraint_index, constraint_index, -1.0);
        for (unsigned int j = 0; j < inputConstraints[i].size() - 1; j++) {
            reluplex.initializeCell(
                    constraint_index,
                    nodeToVars[Index(0, j)],
                    inputConstraints[i][j]);
        }
        reluplex.initializeCell(
                constraint_index,
                constantVar,
                inputConstraints[i].back());
    }

    for (unsigned l = 0; l < transition_layers.size(); l++) {
        auto &layer = transition_layers[l];
        if (l < (transition_layers.size() - 1)) {
            // Set constraints for non-final outputs.
            for (unsigned i = 0; i < safeConstraints.size(); i++) {
                unsigned constraint_index = running_index++;
                reluplex.markBasic(constraint_index); // ?
                reluplex.setUpperBound(constraint_index, 0.0);
                reluplex.initializeCell(constraint_index, constraint_index, -1.0);
                for (unsigned int j = 0; j < safeConstraints[i].size() - 1; j++) {
                    reluplex.initializeCell(
                            constraint_index,
                            nodeToVars[Index(layer, j)],
                            safeConstraints[i][j]);
                }
                reluplex.initializeCell(
                        constraint_index,
                        constantVar,
                        safeConstraints[i].back());
            }
        } else {
            // Set constraints for final outputs.
            for (unsigned i = 0; i < outputConstraints.size(); i++) {
                unsigned constraint_index = running_index++;
                reluplex.markBasic(constraint_index); // ?
                reluplex.setUpperBound(constraint_index, 0.0);
                reluplex.initializeCell(constraint_index, constraint_index, -1.0);
                for (unsigned int j = 0; j < outputConstraints[i].size() - 1; j++) {
                    reluplex.initializeCell(
                            constraint_index,
                            nodeToVars[Index(layer, j)],
                            outputConstraints[i][j]);
                }
                reluplex.initializeCell(
                        constraint_index,
                        constantVar,
                        outputConstraints[i].back());
            }
        }
    }

    reluplex.setLogging(false);
    reluplex.setDumpStates(false);
    reluplex.toggleAlmostBrokenReluEliminiation(false);

    timeval start = Time::sampleMicro();
    timeval end;

    try
    {
        Vector<double> inputs;
        Vector<double> outputs;

        double totalError = 0.0;

        reluplex.setLowerBound(nodeToAux[Index(3, 0)], 0.0);
        reluplex.setUpperBound(nodeToAux[Index(3, 0)], 0.0);
        reluplex.initialize();

        Reluplex::FinalStatus result = reluplex.solve();
        if ( result == Reluplex::SAT )
        {
            out_file << "SAT";
            printf( "Solution found!\n\n" );
            for ( unsigned i = 0; i < 2; ++i )
            {
                double assignment = reluplex.getAssignment(nodeToVars[Index(0, i)]);
                printf("input[%u] = %lf\n", i, assignment);
            }
            for (unsigned li = 0; li < 10; li++) {
                printf("%d[0] (%u) = %lf\n", li, nodeToVars[Index(li, 0)], reluplex.getAssignment(nodeToVars[Index(li, 0)]));
                if (li == 2 || li == 9) {
                    printf("%d[1] (%u) = %lf\n", li, nodeToVars[Index(li, 1)], reluplex.getAssignment(nodeToVars[Index(li, 1)]));
                }
                if (li == 1 || li == 3) {
                    printf("%d[a0] (%u) = %lf\n", li, nodeToAux[Index(li, 0)], reluplex.getAssignment(nodeToAux[Index(li, 0)]));
                }
            }

            //printf( "\n" );
            //for ( unsigned i = 0; i < outputLayerSize; ++i )
            //{
            //    printf( "output[%u] = %.10lf. Normalized: %lf\n", i,
            //            reluplex.getAssignment( nodeToVars[Index(numLayersInUse - 1, i, false)] ),
            //            normalizeOutput( reluplex.getAssignment( nodeToVars[Index(numLayersInUse - 1, i, false)] ),
            //                             neuralNetwork ) );
            //}

            //printf( "\nOutput using nnet:\n" );

            //neuralNetwork.evaluate( inputs, outputs, outputLayerSize );
            //unsigned i = 0;
            //for ( const auto &output : outputs )
            //{
            //    printf( "output[%u] = %.10lf. Normalized: %lf\n", i, output,
            //            normalizeOutput( output, neuralNetwork ) );

            //    totalError +=
            //        FloatUtils::abs( output -
            //                         reluplex.getAssignment( nodeToVars[Index(numLayersInUse - 1, i, false)] ) );

            //    ++i;
            //}

            printf( "\n" );
            printf( "Total error: %.10lf. Average: %.10lf\n", totalError, totalError / 2.0 );
            printf( "\n" );
        }
        else if ( result == Reluplex::UNSAT )
        {
            out_file << "UNS";
            printf( "Can't solve!\n" );
        }
        else if ( result == Reluplex::ERROR )
        {
            printf( "Reluplex error!\n" );
        }
        else
        {
            printf( "Reluplex not done (quit called?)\n" );
        }

        printf( "Number of explored states: %u\n", reluplex.numStatesExplored() );
    }
    catch ( const Error &e )
    {
        printf( "main.cpp: Error caught. Code: %u. Errno: %i. Message: %s\n",
                e.code(),
                e.getErrno(),
                e.userMessage() );
        fflush( 0 );
    }

    end = Time::sampleMicro();

    unsigned milliPassed = Time::timePassed( start, end );
    unsigned seconds = milliPassed / 1000;
    unsigned minutes = seconds / 60;
    unsigned hours = minutes / 60;
    out_file << "," << (((double)milliPassed) / 1000) << std::endl;
    out_file.close();

    printf("Total run time: %u milli (%02u:%02u:%02u)\n",
            Time::timePassed( start, end ), hours, minutes - ( hours * 60 ),
            seconds - ( minutes * 60 ) );

	return 0;
}
