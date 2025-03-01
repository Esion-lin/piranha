#include <iostream>
#include <string>
#include "mpc/maliciously_aby.h"
#include "mpc/swift.h"
#include "globals.h"
#include "mpc/AESObject.h"
#include "mpc/Precompute.h"
#include "util/connect.h"
#include "nn/NeuralNetConfig.h"
#include "nn/NeuralNetwork.h"
#include "test/unitTests.h"
#include "util/Profiler.h"
#include "util/model.h"
#include "nn/MaxpoolLayer.h"
#include "nn/ReLULayer.h"
#include "mpc/RSS.h"
#include "mpc/TPC.h"
#include "mpc/FPC.h"
#include "mpc/OPC.h"
#include "util/util.cuh"
#include "../ext/cxxopts.hpp"
#include <json.hpp>
#include "mpc/Verfiy.h"
#include "mpc/sign.h"
int partyNum;
std::vector<AESObject*> aes_objects;
//AESObject* aes_indep;
//AESObject* aes_next;
//AESObject* aes_prev;
Precompute PrecomputeObject;

extern std::string *addrs;
extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;

extern Profiler matmul_profiler;
Profiler func_profiler;
Profiler memory_profiler;
Profiler comm_profiler;
Profiler debug_profiler;

nlohmann::json piranha_config;

size_t db_bytes = 0;
size_t db_layer_max_bytes = 0;
size_t db_max_bytes = 0;

size_t train_dataset_size = 60000;
size_t test_dataset_size = 10000;
int log_learning_rate = 5;
size_t INPUT_SIZE;
size_t NUM_CLASSES;

void printUsage(const char *bin);
template<typename T, template<typename, typename...> typename Share>
void train(NeuralNetwork<T, Share> *, NeuralNetConfig *config, std::string, std::ifstream &, std::ifstream &, std::ifstream &, std::ifstream &, std::vector<int> &);
template<typename T, template<typename, typename...> typename Share>
void test(NeuralNetwork<T, Share> *, std::ifstream &, std::ifstream &);
void getBatch(std::ifstream &, std::istream_iterator<double> &, std::vector<double> &);
void deleteObjects();
void print_list(std::vector<double> list){
    for(auto ele : list){
        std::cout<<ele<<",";
    }
    std::cout<<"\n";
}
int main(int argc, char** argv) {

    // Parse options -- retrieve party id and config JSON
    cxxopts::Options options("piranha", "GPU-accelerated platform for MPC computation");
    options.add_options()
        ("p,party", "Party number", cxxopts::value<int>())
        ("c,config", "Configuration file", cxxopts::value<std::string>())
        ;
    options.allow_unrecognised_options();

    auto parsed_options = options.parse(argc, argv);

    // Print help
    if (parsed_options.count("help")) {
        std::cout << options.help() << std::endl;
        std::cout << "Report bugs to jlw@berkeley.edu" << std::endl;
        return 0;
    }

    partyNum = parsed_options["party"].as<int>();

    std::ifstream input_config(parsed_options["config"].as<std::string>());
    input_config >> piranha_config;

    // Start memory profiler and initialize communication between parties
    memory_profiler.start();

    //XXX initializeCommunication(options.ip_file, partyNum);
    std::vector<std::string> party_ips;
    for (int i = 0; i < piranha_config["num_parties"]; i++) {
	    party_ips.push_back(piranha_config["party_ips"][i]);
    }
    initializeCommunication(party_ips, partyNum, piranha_config["num_parties"]);

    synchronize(10000, piranha_config["num_parties"]); // wait for everyone to show up :)
    
    for (size_t i = 0; i < piranha_config["num_parties"]; i++) {
        // --------------> AES_TODO
        //Get AES strings from file and create vector of AESObjects
        //options.aes_file;
        //aes_objects[i] = new AESObject(options.)
    }
    //aes_indep = new AESObject(options.aes_indep_file);
    //aes_next = new AESObject(options.aes_next_file);
    //aes_prev = new AESObject(options.aes_prev_file);

    // Unit tests

    //test ring multi
    // test_ring_mul();
    // test_bit_operator();
    test_random();
    // exit(0);
    // uint32_t test_len = 1;
    // std::vector<double> aby_time;
    // std::vector<double> aby_verify_time;
    // std::vector<double> swift_off_time;
    // std::vector<double> swift_online_time;
    // std::vector<double> ours;
    // std::vector<double> ours_verify;
    // uint32_t avg_time = 10;
    // uint32_t deep = 1;
    // for(int kk = 14; kk <= 21; kk++){
    //     test_len = (1<<kk)/deep;
    //     std::vector<double> in_1(test_len), in_2(test_len);
    //     std::vector<double> in_3(test_len);
    //     MSS_Single<uint64_t> a(test_len), b(test_len), c(test_len);
    //     DeviceData<uint64_t> result(a.r_1.size());
    //     std::chrono::steady_clock::time_point start,end;
    //     std::chrono::duration<double> time_span1;

    //     SwiftShareType<uint64_t> swift_a(test_len);
    //     SwiftShareType<uint64_t> swift_b(test_len);
    //     SwiftShareType<uint64_t> swift_c(test_len);
        
    //     RSS<uint64_t> rss_a (in_1, false); 
    //     RSS<uint64_t> rss_b (in_2, false);
    //     RSS<uint64_t> rss_c (in_3, false);


    //     //test aby3

        
    //     func_profiler.clear();
    //     func_profiler.start();
        
    //     double temp_count = 0;
    //     double temp_count2 = 0;

    //     for(int i = 0; i < avg_time*deep; i++){
    //         start= std::chrono::steady_clock::now();
    //         rss_c.zero();
    //         rss_c += rss_a;
    //         rss_c *= rss_b;
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------aby multiplication-------------- cost "<<temp_count/avg_time<<std::endl;
    //     aby_time.push_back(temp_count/avg_time);
    //     temp_count = 0;
    //     for(int i = 0; i < avg_time*deep; i++){
    //         start= std::chrono::steady_clock::now();
    //         RSS_verify<uint64_t>(&rss_a, &rss_b, &rss_c);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------aby multiplication verify-------------- cost "<<temp_count/avg_time<<std::endl;
    //     aby_verify_time.push_back(temp_count/avg_time);
    //     temp_count = 0;
    //     temp_count2 = 0;
    //     for(int i = 0; i < avg_time*deep; i++){
    //         Swift<uint64_t> swift;
    //         start= std::chrono::steady_clock::now();
    //         swift.set_up(&swift_a, &swift_b, &swift_c);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //         start= std::chrono::steady_clock::now();
    //         swift.online(&swift_a, &swift_b, &swift_c);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count2 += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------swift multiplication offline-------------- cost "<<temp_count/avg_time<<std::endl;
    //     std:: cout <<std::endl<<"------------swift multiplication online-------------- cost "<<temp_count2/avg_time<<std::endl;
    //     swift_off_time.push_back(temp_count/avg_time);
    //     swift_online_time.push_back(temp_count2/avg_time);
    //     temp_count = 0;
    //     temp_count2 = 0;
    //     for(int i = 0; i < avg_time*deep; i++){
    //         MSS_Multiplication<uint64_t> mult;
    //         start= std::chrono::steady_clock::now();
    //         mult.set_up(a, b, c);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //         start= std::chrono::steady_clock::now();
    //         mult.online(a, b, c);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count2 += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------mss multiplication setup-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //     std:: cout <<std::endl<<"------------mss multiplication online-------------- cost "<<" "<<temp_count2/avg_time<<std::endl;
    //     double minnn_v = 100000.0;
    //     ours.push_back((temp_count+temp_count2)/avg_time);
    //     MSS  x(test_len*deep), y(test_len*deep), z(1);
    //     if(kk<=15){
    //         temp_count = 0;
    //         for(int i = 0; i < avg_time; i++){
                
    //             start= std::chrono::steady_clock::now();
    //             DotVerifyWithReduce<3>(&x, &y, &z);
    //             end = std::chrono::steady_clock::now();
    //             time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //             temp_count += time_span1.count();
    //         }
    //         std:: cout <<std::endl<<"------------mss multiplication verify-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //         if(temp_count/avg_time < minnn_v)
    //             minnn_v = temp_count/avg_time;
    //     }
    //     if(kk<=16){
    //         temp_count = 0;
    //         for(int i = 0; i < avg_time; i++){
    //             start= std::chrono::steady_clock::now();
    //             DotVerifyWithReduce<4>(&x, &y, &z);
    //             end = std::chrono::steady_clock::now();
    //             time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //             temp_count += time_span1.count();
    //         }
    //         std:: cout <<std::endl<<"------------mss multiplication verify-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //         if(temp_count/avg_time < minnn_v)
    //             minnn_v = temp_count/avg_time;
    //     }
    //     if(kk<=17){
    //         temp_count = 0;
    //         for(int i = 0; i < avg_time; i++){
    //             start= std::chrono::steady_clock::now();
    //             DotVerifyWithReduce<5>(&x, &y, &z);
    //             end = std::chrono::steady_clock::now();
    //             time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //             temp_count += time_span1.count();
    //         }
    //         std:: cout <<std::endl<<"------------mss multiplication verify-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //         if(temp_count/avg_time < minnn_v)
    //             minnn_v = temp_count/avg_time;
    //     }
    //     temp_count = 0;
    //     for(int i = 0; i < avg_time; i++){
    //         start= std::chrono::steady_clock::now();
    //         DotVerifyWithReduce<6>(&x, &y, &z);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------mss multiplication verify-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //     if(temp_count/avg_time < minnn_v)
    //             minnn_v = temp_count/avg_time;

    //     temp_count = 0;
    //     for(int i = 0; i < avg_time; i++){
    //         start= std::chrono::steady_clock::now();
    //         DotVerifyWithReduce<7>(&x, &y, &z);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------mss multiplication verify-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //     if(temp_count/avg_time < minnn_v)
    //             minnn_v = temp_count/avg_time;

    //     temp_count = 0;
    //     for(int i = 0; i < avg_time; i++){
    //         start= std::chrono::steady_clock::now();
    //         DotVerifyWithReduce<8>(&x, &y, &z);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------mss multiplication verify-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //     if(temp_count/avg_time < minnn_v)
    //             minnn_v = temp_count/avg_time;

    //     temp_count = 0;
    //     for(int i = 0; i < avg_time; i++){
    //         start= std::chrono::steady_clock::now();
    //         DotVerifyWithReduce<9>(&x, &y, &z);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------mss multiplication verify-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //     if(temp_count/avg_time < minnn_v)
    //             minnn_v = temp_count/avg_time;

    //     temp_count = 0;
    //     for(int i = 0; i < avg_time; i++){
    //         start= std::chrono::steady_clock::now();
    //         DotVerifyWithReduce<10>(&x, &y, &z);
    //         end = std::chrono::steady_clock::now();
    //         time_span1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start);
    //         temp_count += time_span1.count();
    //     }
    //     std:: cout <<std::endl<<"------------mss multiplication verify-------------- cost "<<" "<<temp_count/avg_time<<std::endl;
    //     if(temp_count/avg_time < minnn_v)
    //             minnn_v = temp_count/avg_time;
    //     ours_verify.push_back(minnn_v);
    // }
    // printf("aby cost\n");
    // print_list(aby_time);
    // printf("aby verify cost\n");
    // print_list(aby_verify_time);
    // printf("swift offline cost\n");
    // print_list(swift_off_time);
    // printf("swift online cost\n");
    // print_list(swift_online_time);
    // printf("ours cost\n");
    // print_list(ours);
    // printf("ours verify cost\n");
    // print_list(ours_verify);
    
    exit(0);
    
}

template<typename T, template<typename, typename...> typename Share>
void updateAccuracy(NeuralNetwork<T, Share> *net, std::vector<double> &labels, int &correct) {

    Share<T> *activations = net->layers[net->layers.size() - 1]->getActivation();
    //printShareFinite(*activations, "last layer activations", 10);

    DeviceData<T> reconstructedOutput(activations->size());
    reconstruct(*activations, reconstructedOutput);

    std::vector<double> hostOutput(reconstructedOutput.size());
    copyToHost(reconstructedOutput, hostOutput, true);

    int nClasses = hostOutput.size() / MINI_BATCH_SIZE;

    for(int i = 0; i < MINI_BATCH_SIZE; i++) {
        auto result = std::max_element(hostOutput.begin() + (i * nClasses), hostOutput.begin() + ((i+1) * nClasses));
        int max_index = std::distance(hostOutput.begin(), result);

        if (labels[max_index] == 1.0) {
            correct++;
        }
    }
}

template<typename T, template<typename, typename...> typename Share>
void test(NeuralNetwork<T, Share> *net, std::ifstream &test_data, std::ifstream &test_labels) {

    std::istream_iterator<double> data_it(test_data);
    std::istream_iterator<double> label_it(test_labels);

    if (piranha_config["debug_print"]) {
        std::cout << std::endl << " == Testing == " << std::endl << std::endl;
    }

    size_t numIterations = test_dataset_size / MINI_BATCH_SIZE;
    int correct = 0;

    for (int i = 0; i < numIterations; i++) {

        std::vector<double> batch_data(MINI_BATCH_SIZE * INPUT_SIZE);
        std::vector<double> batch_labels(MINI_BATCH_SIZE * NUM_CLASSES);

        getBatch(test_data, data_it, batch_data);
        getBatch(test_labels, label_it, batch_labels);

        net->forward(batch_data);

        updateAccuracy(net, batch_labels, correct);
    }

    double acc = ((double)correct) / (numIterations * MINI_BATCH_SIZE);

    if (piranha_config["eval_accuracy"]) {
        printf("test accuracy,%f\n", acc);
    }
}

template<typename T, template<typename, typename...> typename Share>
void train(NeuralNetwork<T, Share> *net, NeuralNetConfig *config, std::string run_name,
        std::ifstream &train_data, std::ifstream &train_labels,
        std::ifstream &test_data, std::ifstream &test_labels,
        std::vector<int> &learning_rate) {

    size_t numIterations = (train_dataset_size / MINI_BATCH_SIZE); //+ 1; // assumes dataset doesn't divide equally into batches
    if (piranha_config["custom_iterations"]) {
        numIterations = (size_t) piranha_config["custom_iteration_count"];
    }

    size_t numEpochs = learning_rate.size(); // learning_rate.size() / ((numIterations / iterations_per_lr) + 1);
    if (piranha_config["custom_epochs"]) {
        numEpochs = (size_t) piranha_config["custom_epoch_count"];
    } 

    printf("TRAINING, EPOCHS = %d ITERATIONS = %d\n", numEpochs, numIterations);

    std::istream_iterator<double> data_it(train_data);
    std::istream_iterator<double> label_it(train_labels);

    if (piranha_config["debug_print"]) {
        std::cout << std::endl << " == Training (" << numEpochs << " epochs) ==" << std::endl;
    }

    int lr_ctr = 0;

    double total_comm_tx_mb = 0.0;
    double total_comm_rx_mb = 0.0;
    double total_time_s = 0.0;

    for (int e = 0; e < numEpochs; e++) {

        if (piranha_config["debug_print"]) {
            std::cout << std::endl << " -- Epoch " << e << " (" << numIterations << " iterations, log_lr = " << learning_rate[e] << ") --" << std::endl;
        }
        log_learning_rate = learning_rate[e];

        int correct = 0;
        for (int i = 0; i < numIterations; i++) {

            comm_profiler.clear();

            if (piranha_config["debug_print"]) {
                printf("iteration,%d\n", i);
            }

            std::vector<double> batch_data(MINI_BATCH_SIZE * INPUT_SIZE);
            std::vector<double> batch_labels(MINI_BATCH_SIZE * NUM_CLASSES);

            getBatch(train_data, data_it, batch_data);
            getBatch(train_labels, label_it, batch_labels);

            Profiler toplevel_profiler;

            toplevel_profiler.start();
            net->forward(batch_data);
            toplevel_profiler.accumulate("fw-pass");

            updateAccuracy(net, batch_labels, correct);

            if (piranha_config["eval_inference_stats"]) {
                double fw_ms = toplevel_profiler.get_elapsed("fw-pass");
                
                printf("inference iteration (ms),%f\n", fw_ms);
                printf("inference TX comm (bytes),%d\n", comm_profiler.get_comm_tx_bytes());
                printf("inference RX comm (bytes),%d\n", comm_profiler.get_comm_rx_bytes());
            }

            if (piranha_config["eval_fw_peak_memory"]) {
                printf("fw pass peak memory (MB), %f\n", memory_profiler.get_max_mem_mb());
            }

            total_comm_tx_mb += ((double)comm_profiler.get_comm_tx_bytes()) / 1024.0 / 1024.0;
            total_comm_rx_mb += ((double)comm_profiler.get_comm_rx_bytes()) / 1024.0 / 1024.0;

            if (piranha_config["inference_only"]) {
                continue;
            }

            toplevel_profiler.start();
            net->backward(batch_labels);
            toplevel_profiler.accumulate("bw-pass");

            if (piranha_config["eval_train_stats"]) {
                double fw_bw_ms = toplevel_profiler.get_elapsed_all();
                printf("training iteration (ms),%f\n", fw_bw_ms);
                printf("training TX comm (MB),%f\n", comm_profiler.get_comm_tx_bytes() / 1024.0 / 1024.0);
                printf("training RX comm (MB),%f\n", comm_profiler.get_comm_rx_bytes() / 1024.0 / 1024.0);

                double comm_ms = comm_profiler.get_elapsed("comm-time");
                printf("training comm (ms),%f\n", comm_ms);
                printf("training computation (ms),%f\n", fw_bw_ms - comm_ms);
            }

            if (piranha_config["eval_bw_peak_memory"]) {
                printf("total fw-bw pass peak memory (MB), %f\n", memory_profiler.get_max_mem_mb());
            }

            total_time_s += toplevel_profiler.get_elapsed_all() / 1000.0;

            
            if (piranha_config["iteration_snapshots"]) {
                std::string snapshot_path = "output/"+run_name+"-epoch-"+std::to_string(e)+"-iteration-"+std::to_string(i);
                net->saveSnapshot(snapshot_path);

                if (piranha_config["test_iteration_snapshots"]) {
                    NeuralNetwork<uint64_t, OPC> test_net(config, 0);
                    test_net.loadSnapshot(snapshot_path);

                    test(&test_net, test_data, test_labels);
                    fflush(stdout);
                }
            }
        }

        if (piranha_config["eval_epoch_stats"]) {
            printf("epoch,%d\n", e);
            printf("total time (s),%f\n", total_time_s);
            printf("total tx comm (MB),%f\n", total_comm_tx_mb);
            printf("total rx comm (MB),%f\n", total_comm_rx_mb);
        }

        double acc = ((double)correct) / (numIterations * MINI_BATCH_SIZE);
        if (piranha_config["eval_accuracy"]) {
            printf("train accuracy,%f\n", acc);
        }

        if (!piranha_config["no_test"]) {
            test(net, test_data, test_labels);
        }

        if (piranha_config["last_test"] && e == numEpochs - 1) {
            test(net, test_data, test_labels);
        }

        if (piranha_config["epoch_snapshots"]) {
            net->saveSnapshot("output/"+run_name+"-epoch-"+std::to_string(e));
        }

        fflush(stdout);
    }
}

void getBatch(std::ifstream &f, std::istream_iterator<double> &it, std::vector<double> &vals) {
    
    std::istream_iterator<double> eos;
    for(int i = 0; i < vals.size(); i++) {
        if (it == eos) {
            f.clear();
            f.seekg(0);
            std::istream_iterator<double> new_it(f);    
            it = new_it;
        }

        vals[i] = *it++;
    }
}

void deleteObjects() {
	//close connection
	for (int i = 0; i < piranha_config["num_parties"]; i++) {
		if (i != partyNum) {
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;
	delete[] addrs;
}

