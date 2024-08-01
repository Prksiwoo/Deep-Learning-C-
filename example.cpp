#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cmath>

class Node {
    public:
        int node_number = 0; //number representing this node.
        std::unordered_map<int, float> pointing_weights = {}; //map representing weight pointing from this node to other nodes. key: node number, value: weight value.
        std::unordered_map<int, float> pointed_weights = {}; //map representing weight pointed to this node from other nodes. key: node number, value: weight value.
        float bias = 0; //bias of this node.
        float (*activation)(float); //function pointer representing activation function for each different nodes.
        float (*activation_prime)(float); // function pointer representing derivative of activation function.
        float activate(float input) {
            return activation(input);
        };
        float activate_prime(float input) {
            return activation_prime(input);
        };
};

class Model {
    public:
        int number_of_inputs = 0; //number of input data points.
        int number_of_outputs = 0; //number of output data points.
        std::unordered_map<int, Node> nodes = {}; //map representing all the nodes inside this model. key: node number, value: node.
        float (*loss_function_prime)(int, float, float); // function pointer representing partial derivative (by specific output) of loss function.
        float loss_prime(int number_of_outputs, float actual_output, float training_data) {
            return loss_function_prime(number_of_outputs, actual_output, training_data);
        };
        //make connection between 2 nodes
        bool connect(int from, int to, float weight) {
            //from: node number of node that is connecting from.
            //to: node number of node that is being connected to.
            //weight: weight value.

            //assign value.
            nodes[from].pointing_weights[to] = weight;
            nodes[to].pointed_weights[from] = weight;
        };

        //add node and returns node number
        int add_node(float bias) {

            int size = nodes.size();

            //make new instance of node.
            Node node = {};
            node.node_number = size;
            node.bias = bias;

            //assign value.
            nodes[size] = node;
            
            return size;
        };
        //forward propagate via model
        //recursive function
        /*
        description
            for many unknown nodes, find any unknown node whose key of pointed weights are subset of known nodes.
            calculate the value of that unknown node and add that node to known nodes.
            iterate that over till all the output nodes be known.
            knowing means getting the output value of each node. this is different from output of model.
        */
        //pair.first: outputs, pair.second: nodes calculated values
        std::vector<float> run(std::vector<float> inputs) {
            //returns only outputs

            //declaring instnaces for recursive forward propagation
            std::unordered_map<int, std::pair<float, float>> empty = {};
            std::unordered_set<int> set = {};
            for (int i = 0; i < nodes.size(); i++) {
                set.insert(i);
            };
            std::unordered_map<int, std::pair<float, float>> result = {};
            result = forward_propagate(inputs, empty, set);
            std::vector<float> outputs = {};
            for (int i = number_of_inputs; i < number_of_inputs + number_of_outputs; i++) {
                //iterating over output nodes
                outputs.push_back(result[i].second);
            };
            return outputs;
        };
        bool train(std::pair<std::vector<float>, std::vector<float>> dataset, float learning_rate) {
            //dataset.first: input vector
            //dataset.second: output vector

            //declaring instances for recursive forward propagation
            std::unordered_map<int, std::pair<float, float>> empty = {};
            std::unordered_set<int> set = {};
            for (int i = 0; i < nodes.size(); i++) {
                set.insert(i);
            };
            std::unordered_map<int, std::pair<float, float>> result = {};
            result = forward_propagate(dataset.first, empty, set);

            //declaring empty inputs for recursive back propagation
            std::unordered_map<int, float> empty_gradients = {};
            return back_propagate(result, empty_gradients, set, dataset.second, learning_rate);
        };
    private:
            std::unordered_map<int, std::pair<float, float>> forward_propagate(std::vector<float> inputs, std::unordered_map<int, std::pair<float, float>> calculated_nodes_values, std::unordered_set<int> unknown_nodes) {
                //inputs: vector of input values.
                //calculated_nodes_value: output value of each known node. key: node number, value.first: output value of the node before activation, value.second: after activation.

                //exit condition
                if (unknown_nodes.size() == 0) {
                    //declaring result
                    std::unordered_map<int, std::pair<float, float>> result = {};
                    result = calculated_nodes_values;
                    return result;
                };
                //for the first recursive call, add input nodes to calculated_nodes_values map.
                if (calculated_nodes_values.size() == 0) {
                    for (int i = 0; i < number_of_inputs; i++) {
                        //i: input node number
                        calculated_nodes_values[i].first = 0;
                        //input node's value before activation is undefined.
                        //assigning dummy value for simpler back propagation.

                        calculated_nodes_values[i].second = inputs[i];
                        unknown_nodes.erase(i);
                    };
                };

                //updating calculated_nodes_values map.
                for (int node : unknown_nodes) {
                    //node: unknown node number

                    //check if all nodes that are pointing to this node is subset of found nodes.
                    bool is_subset = true;
                    for (const auto& pair : nodes[node].pointed_weights) {
                        if (calculated_nodes_values.count(pair.first) == 0) {
                            is_subset = false;
                            break;
                        };
                    };
                    if (is_subset == false) {
                        continue;
                    };

                    //set new value of the node
                    float value = 0.0f;
                    for (const auto& pair : nodes[node].pointed_weights) {
                        //weighted sum
                        value += nodes[node].pointed_weights[pair.first] * calculated_nodes_values[pair.first].first;
                    };
                    //adding bias
                    value = value + nodes[node].bias;
                    //pushing new value into calculated_nodes_values
                    calculated_nodes_values[node].first = value;
                    calculated_nodes_values[node].second = nodes[node].activate(value);
                    unknown_nodes.erase(node);
                    break;
                };
                return forward_propagate(inputs, calculated_nodes_values, unknown_nodes);
            };

            bool back_propagate(std::unordered_map<int, std::pair<float, float>> result, std::unordered_map<int, float> calculated_nodes_gradients, std::unordered_set<int> unknown_nodes, std::vector<float> training_output_data, float learning_rate) {
                //find the node whose pointing weights are subset of known node
                //initially output node is known
                //result.first: node value before activation
                //result.second: node value after activation

                //initialize gradients to output nodes for the first recursive call
                    if (calculated_nodes_gradients.size() == 0) {
                    for (int i = number_of_inputs; i < number_of_inputs + number_of_outputs; i++) {
                        //iterating over output nodes
                        calculated_nodes_gradients[i] = loss_prime(number_of_outputs, result[i].second, training_output_data[i - number_of_inputs]) * nodes[i].activate_prime(result[i].first);
                        //reason why using training_output_data[i - number_of_inputs]) is training_output_data is an vector initialized from 0.
                        //to get the optimal training_output_data by output node number.
                        unknown_nodes.erase(i);
                    };
                };

                //exit condition
                if (unknown_nodes.size() == 0) {
                //adjusting biases
                    for (int i = number_of_inputs; i < nodes.size(); i++) {
                        //iterate over nodes excluding input nodes
                        float dldb = calculated_nodes_gradients[i];
                        float new_bias = nodes[i].bias - learning_rate * dldb;
                        nodes[i].bias = new_bias;
                    };
                    return true;
                };

                for (int node : unknown_nodes) {
                    //check if all nodes that are pointed from this node is subset of found nodes.
                    bool is_subset = true;
                    for (const auto& pair : nodes[node].pointing_weights) {
                        if (calculated_nodes_gradients.count(pair.first) == 0) {
                            is_subset = false;
                            break;
                        };
                    };
                    if (is_subset == false) {
                        continue;
                    };

                    //weight adjusting and gradient assigning
                    float gradient = 0.0f;
                    for (const auto& pair : nodes[node].pointing_weights) {
                        //dldw: derivative of loss function respect to weight
                        float dldw = calculated_nodes_gradients[pair.first] * result[node].second;
                        float new_weight = pair.second - learning_rate * dldw;
                        connect(node, pair.first, new_weight);
                        gradient = gradient + calculated_nodes_gradients[pair.first] * new_weight * nodes[node].activate_prime(result[node].first);
                    };
                    calculated_nodes_gradients[node] = gradient;
                    unknown_nodes.erase(node);
                    break;
                };
                back_propagate(result, calculated_nodes_gradients, unknown_nodes, training_output_data, learning_rate);
            };
};

//returns default model with input, output node labed. default node starts with node number (number of inputs + number of outputs)
Model new_model(int number_of_inputs, int number_of_outputs) {
    //declare a map of nodes instance
    std::unordered_map<int, Node> nodes = {};

    //iterate over number of inputs and number each nodes
    for (int i = 0; i < number_of_inputs; i++) {
        Node node {
            .node_number = i,
        };
        nodes[i] = node;
    };

    //iterate over number of outputs and number each nodes
    for (int i = number_of_inputs; i < number_of_inputs + number_of_outputs; i++) {
        Node node {
            .node_number = i,
        };
        nodes[i] = node;
    };

    //now the nodes (whose node number >= number of inputs + number of outputs) are non-input non-output nodes.
    //assign that map of nodes to model instance.
    Model model {
        .number_of_inputs = number_of_inputs,
        .number_of_outputs = number_of_outputs,
        .nodes = nodes,
    };

    //return the final model
    return model;
};

float sigmoid(float input) {
    //calculating denominator
    float denominator = 1 + std::exp(-1 * input);
    return 1 / denominator;
};

float sigmoid_derivative(float input) {
    float value = sigmoid(input);
    return value * (1 - value);
};

float mean_square_error_derivative(int number_of_outputs, float actual_output, float training_output_data) {
    float value = 2 * actual_output - 2 * training_output_data;
    return value / number_of_outputs;
};

int main() {
    Model model = new_model(2, 1);
    model.add_node(0.3f);
    model.add_node(0.6f);
    model.nodes[2].bias = 0.8f;
    model.connect(0, 3, 0.3f);
    model.connect(1, 3, 0.2f);
    model.connect(3, 4, 0.6f);
    model.connect(0, 2, 0.4f);
    model.connect(4, 2, 0.7f);
    std::vector<float> inputs = {};
    inputs.push_back(0.5f);
    inputs.push_back(0.4f);
    for (int i = 0; i < model.nodes.size(); i++) {
        model.nodes[i].activation = sigmoid;
        model.nodes[i].activation_prime = sigmoid_derivative;
    };
    model.loss_function_prime = mean_square_error_derivative;

    std::vector<float> output = model.run(inputs);
    std::cout << output[0] << std::endl;

    std::pair<std::vector<float>, std::vector<float>> dataset = {};
    dataset.first = inputs;
    dataset.second.push_back(0.5);
    float error_rate = 10;
    int iter = 0;
    while (error_rate > 0.0000001) {
        iter++;
        model.train(dataset, 0.1f);
        output = model.run(inputs);
        float value = (output[0] - 0.5);
        error_rate = value * value;
    };
    std::cout << output[0] << std::endl;
    std::cout << iter << std::endl;
    std::cout << error_rate << std::endl;
    return 0;
};