#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <getopt.h>
#include <filesystem>
#include <unistd.h>
#include <limits.h>

#include "tbb/parallel_for_each.h"
#include "tbb/task_scheduler_init.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <unistd.h>


#include "ExaTrkXTriton.hpp"
#include "ExaTrkXTrackFindingTriton.hpp"

#include <string>
#include <vector>
#include <memory>

namespace fs = std::filesystem;

void processInput(std::string file_path, std::vector<float>& input_tensor_values){
    input_tensor_values.clear();

    std::ifstream f (file_path);   /* open file */
    if (!f.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + file_path).c_str());
    }
    std::string line;                    /* string to hold each line */
    while (getline (f, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        //array.push_back (row);          /* add row to array */
        input_tensor_values.insert (input_tensor_values.end(),row.begin(),row.end());  
    }
    f.close();
}

void dumpTrackCandidate(const std::vector<std::vector<int> >& trackCandidates) {
    int idx = 0;
    for (const auto& track_candidate : trackCandidates) {
        std::cout << "Track candidate: " << idx++ << "--> ";
        for (const auto& id : track_candidate) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
    }
}

std::string get_selfpath() {
    char buff[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
    if (len != -1) {
      buff[len] = '\0';
      std::string bin_path = buff;

      int pos_to_erase = bin_path.find("build/bin/inference");
      if (pos_to_erase != std::string::npos) {
        std::string folder_path = bin_path.erase(pos_to_erase, 19);
        return folder_path;
      } else {
        /* handle error condition */
        return "";
      }
    } else {
        /* handle error condition */
        return "";
    }
}

// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    std::string folder_path = get_selfpath();   
    int server_type = 0;
    std::string input_file_path = folder_path + "datanmodels/in_e1000.csv";
    int opt;
    bool help = false;
    bool verbose = false;
    int nthreads = 1;
    std::string url("localhost:8001");
    std::string triton_model_name("exatrkxcpu");
    
    while ((opt = getopt(argc, argv, "vht:d:u:m:")) != -1) {
        switch (opt) {
            case 'd':
                input_file_path = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            case 't':
                nthreads = atoi(optarg);
                break;
            case 'u':
                url = optarg;
                break;
            case 'm':
                triton_model_name = optarg;
                break;
            case 'h':
                help = true;
            default:
                fprintf(stderr, "Usage: %s [-hv] [-m triton_model_name] [-d input_file_path] [-s server_type]\n", argv[0]);
                if (help) {
                    std::cerr << " -m: triton model name, exatrkxcpu/exatrkxgpu" << std::endl;
                    std::cerr << " -d: input data/directory" << std::endl;
                    std::cerr << " -t: number of threads" << std::endl;
                    std::cerr << " -u: url of server" << std::endl;
                    std::cerr << " -v: verbose" << std::endl;
                }
            exit(EXIT_FAILURE);
        }
    }

    // start tbb scheduler
    tbb::task_scheduler_init init(nthreads);

    std::cout << "Input file: " << input_file_path << std::endl;
    

    std::unique_ptr<ExaTrkXTrackFindingTriton> infer;
    ExaTrkXTrackFindingTriton::Config config{triton_model_name, url, verbose};
    infer = std::make_unique<ExaTrkXTrackFindingTriton>(config);

    std::cout << "Running Inference with ExaTrkX as a service." << std::endl;
    
    const fs::path filepath(input_file_path);
    std::error_code ec;
    int tot_tracks = 0;
    int ievt = 0;

    auto run_one_file = [&](const fs::path& in_file_name) -> void {
        // read spacepoints table saved in csv
        std::vector<float> input_tensor_values;
        processInput(in_file_name, input_tensor_values);
        int64_t spacepointFeatures = 3;

        int numSpacepoints = input_tensor_values.size()/spacepointFeatures;

        std::vector<int> spacepoint_ids;
        for (int i=0; i < numSpacepoints; ++i){
            spacepoint_ids.push_back(i);
        }
        std::vector<std::vector<int> > track_candidates;
        infer->getTracks(input_tensor_values, spacepoint_ids, track_candidates);
        tot_tracks += track_candidates.size();
        ievt ++;
    };


    if (fs::is_directory(filepath, ec)) {
        if (nthreads > 1) {
            // concurrent execution of all files in directory
            std::vector<std::string>  filenames;
            for (auto& entry : fs::directory_iterator(filepath)) {
                if (fs::is_regular_file(entry.path())) {
                    filenames.push_back(entry.path().string());
                }
            }
            int nfiles = std::distance(filenames.begin(), filenames.end());
            std::cout << "Running " << nfiles << " files in " << nthreads << " threads." << std::endl;

            tbb::parallel_for_each(
                filenames.begin(), filenames.end(),
                [&](const std::string& fname) {
                    run_one_file(fname);
                });  // end parallel_for_each            

        } else {
            // sequential execution of all files in directory
            for(auto& entry : fs::directory_iterator(filepath)) {
                if (fs::is_regular_file(entry.path())) {
                    // std::cout << "Processing file: " << entry.path().string() << std::endl;
                    run_one_file(entry.path().string());
                }
            }
        }

    } else if (fs::is_regular_file(filepath, ec)) {
        run_one_file(filepath);
    } else {
        std::cerr << "Error: " << filepath << " is not a file or directory." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // summarize the total number of tracks.
    std::cout << "Total " << tot_tracks << " tracks in " << ievt << " events" << std::endl;
    return 0;
}
