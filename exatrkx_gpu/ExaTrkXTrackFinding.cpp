#include "ExaTrkXTrackFinding.hpp"
#include "ExaTrkXUtils.hpp"

#include <torch/torch.h>
#include <torch/script.h>
using namespace torch::indexing;

#include <grid.h>
#include <insert_points.h>
#include <counting_sort.h>
#include <prefix_sum.h>
#include <find_nbrs.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
// #include "mmio_read.h"


ExaTrkXTrackFinding::ExaTrkXTrackFinding(
    const ExaTrkXTrackFinding::Config& config): m_cfg(config)
{
    initTrainedModels();
}

void ExaTrkXTrackFinding::initTrainedModels(){
    std::string l_embedModelPath(m_cfg.modelDir + "/torchscript/embed.pt");
    std::string l_filterModelPath(m_cfg.modelDir + "/torchscript/filter.pt");
    std::string l_gnnModelPath(m_cfg.modelDir + "/torchscript/gnn.pt");
    c10::InferenceMode guard(true);
    try {
        std::cout << "Test Point m_cfg in initTrainedModels:  " << m_cfg.device_id << std::endl;
        torch::Device device(torch::kCUDA, m_cfg.device_id);
        e_model = torch::jit::load(l_embedModelPath.c_str(), device);
        e_model.eval();
        f_model = torch::jit::load(l_filterModelPath.c_str(), device);
        f_model.eval();
        g_model = torch::jit::load(l_gnnModelPath.c_str(), device);
        g_model.eval();
    } catch (const c10::Error& e) {
        throw std::invalid_argument("Failed to load models: " + e.msg());
    }
}


// The main function that runs the Exa.TrkX ExaTrkXTrackFindingence pipeline
// Be care of sharpe corners.
void ExaTrkXTrackFinding::getTracks(
    std::vector<float>& inputValues,
    std::vector<int>& spacepointIDs,
    std::vector<std::vector<int> >& trackCandidates,
    ExaTrkXTime& timeInfo,
    int32_t device_id) const {

    ExaTrkXTimer tot_timer;
    tot_timer.start();
    // hardcoded debugging information
    c10::InferenceMode guard(true);
    bool debug = true;
    std::cout << "Test Point 0:  " << device_id << std::endl;
    torch::Device device(torch::kCUDA, device_id);


    // printout the r,phi,z of the first spacepoint
    // std::cout <<"First spacepoint information: ";
    // std::copy(inputValues.begin(), inputValues.begin() + 3,
    //           std::ostream_iterator<float>(std::cout, " "));
    // std::cout << std::endl;

    ExaTrkXTimer timer;
    // ************
    // Embedding
    // ************

    timer.start();
    int64_t numSpacepoints = inputValues.size() / m_cfg.spacepointFeatures;
    std::vector<torch::jit::IValue> eInputTensorJit;
    auto e_opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor eLibInputTensor = torch::from_blob(
        inputValues.data(),
        {numSpacepoints, m_cfg.spacepointFeatures},
        e_opts).to(torch::kFloat32);

    eInputTensorJit.push_back(eLibInputTensor.to(device));
    std::cout <<"Test Point 1:  " << std::endl;
    if (eInputTensorJit[0].isTensor()) {
        torch::Tensor tensor = eInputTensorJit[0].toTensor();
        torch::Device test_tensor_device = tensor.device();

        if (test_tensor_device.is_cpu()) {
            std::cout << "The tensor is on CPU." << std::endl;
        } else if (test_tensor_device.is_cuda()) {
            if (test_tensor_device.has_index()) {
                std::cout << "The tensor is on GPU: " << static_cast<int>(test_tensor_device.index()) << std::endl;
            } else {
                std::cout << "The tensor is on GPU: 0" << std::endl;  // Assuming default GPU index is 0
            }
        }
    } else {
        // Handle the case where the IValue is not a tensor
        std::cerr << "The IValue is not a tensor." << std::endl;
    }
    // auto parameters = e_model.named_parameters();
    // if (parameters.size() > 0) {
    //     auto first_param = parameters.begin();
    //     auto device = first_param.value().device();
    //     std::cout << "The e_model is on GPU: " << device.index() << std::endl;
    // } else {
    //     std::cout << "The e_model has no parameters." << std::endl;
    // }

    // check which GPU the e_model is on
    // auto parameters = e_model.named_parameters();
    // if (parameters.size() > 0) {
    //     auto first_param = parameters.begin();
    //     auto device = first_param.value().device();
    //     std::cout << "The e_model is on GPU: " << device.index() << std::endl;
    // } else {
    //     std::cout << "The e_model has no parameters." << std::endl;
    // }

    std::cout <<" eOutput = e_model.forward(eInputTensorJit).toTensor() " << std::endl;
    at::Tensor eOutput = e_model.forward(eInputTensorJit).toTensor();
    std::cout <<"Embedding space of libtorch the first SP: \n";
    std::cout << eOutput.slice(/*dim=*/0, /*start=*/0, /*end=*/1) << std::endl;
    std::cout << std::endl;

    timeInfo.embedding = timer.stopAndGetElapsedTime();

    // ************
    // Building Edges
    // ************
    timer.start();
    torch::Tensor edgeList = buildEdges(
        eOutput, numSpacepoints, m_cfg.embeddingDim, m_cfg.rVal, m_cfg.knnVal, device_id);
    int64_t numEdges = edgeList.size(1);

    std::cout << "Built " << edgeList.size(1) << " edges. " <<  edgeList.size(0) << std::endl;
    std::cout << edgeList.slice(1, 0, 5) << std::endl;

    timeInfo.building = timer.stopAndGetElapsedTime();

    // ************
    // Filtering
    // ************
    // std::cout << "Get scores for " << numEdges<< " edges." << std::endl;

    timer.start();
    std::vector<torch::jit::IValue> fInputTensorJit;
    fInputTensorJit.push_back(eLibInputTensor.to(device));
    edgeList = edgeList.to(device);
    fInputTensorJit.push_back(edgeList);
    at::Tensor fOutput = f_model.forward(fInputTensorJit).toTensor();
    fOutput.squeeze_();
    fOutput.sigmoid_();

    // std::cout << "After filtering: " << fOutput.size(0) << " " << fOutput.size(1) << std::endl;
    // std::cout << fOutput.slice(/*dim=*/0, /*start=*/0, /*end=*/9) << std::endl;

    torch::Tensor filterMask = fOutput > m_cfg.filterCut;
    torch::Tensor edgesAfterF = edgeList.index({Slice(), filterMask});
    edgesAfterF = edgesAfterF.to(torch::kInt64);
    int64_t numEdgesAfterF = edgesAfterF.size(1);

    // std::cout << "After filtering: " << numEdgesAfterF << " edges." << std::endl;

    timeInfo.filtering = timer.stopAndGetElapsedTime();

    // ************
    // GNN
    // ************
    timer.start();

    std::vector<torch::jit::IValue> gInputTensorJit;
    auto g_opts = torch::TensorOptions().dtype(torch::kInt64);
    gInputTensorJit.push_back(eLibInputTensor.to(device));
    gInputTensorJit.push_back(edgesAfterF.to(device));
    auto gOutput = g_model.forward(gInputTensorJit).toTensor();
    gOutput.sigmoid_();
    gOutput = gOutput.cpu();
    timeInfo.gnn = timer.stopAndGetElapsedTime();

    // std::cout << "GNN scores for " << gOutput.size(0) << " edges." << std::endl;
    // std::cout << gOutput.slice(0, 0, 5) << std::endl;

    // ************
    // Track Labeling with cugraph::connected_components
    // ************
    timer.start();

    using vertex_t = int32_t;
    std::vector<vertex_t> rowIndices;
    std::vector<vertex_t> colIndices;
    std::vector<float> edgeWeights;
    std::vector<vertex_t> trackLabels(numSpacepoints);
    // Move tensor to CPU if it's on another device
    edgesAfterF = edgesAfterF.to(torch::kCPU);
    std::copy(
        edgesAfterF.data_ptr<int64_t>(),
        edgesAfterF.data_ptr<int64_t>()+numEdgesAfterF,
        std::back_insert_iterator(rowIndices));
    std::copy(
        edgesAfterF.data_ptr<int64_t>()+numEdgesAfterF,
        edgesAfterF.data_ptr<int64_t>() + numEdgesAfterF+numEdgesAfterF,
        std::back_insert_iterator(colIndices));
    std::copy(
        gOutput.data_ptr<float>(),
        gOutput.data_ptr<float>() + numEdgesAfterF,
        std::back_insert_iterator(edgeWeights));

    weaklyConnectedComponents<int32_t,int32_t,float>(
        numSpacepoints,
        rowIndices, colIndices, edgeWeights, trackLabels);

    // weakly_connected_components<int32_t,int32_t,float>(
    //     rowIndices, colIndices, edgeWeights, trackLabels);

    // std::cout << "size of components: " << trackLabels.size() << std::endl;
    if (trackLabels.size() == 0)  return;


    trackCandidates.clear();

    int existTrkIdx = 0;
    // map labeling from MCC to customized track id.
    std::map<int32_t, int32_t> trackLableToIds;

    for(int32_t idx=0; idx < numSpacepoints; ++idx) {
        int32_t trackLabel = trackLabels[idx];
        int spacepointID = spacepointIDs[idx];

        int trkId;
        if(trackLableToIds.find(trackLabel) != trackLableToIds.end()) {
            trkId = trackLableToIds[trackLabel];
            trackCandidates[trkId].push_back(spacepointID);
        } else {
            // a new track, assign the track id
            // and create a vector
            trkId = existTrkIdx;
            trackCandidates.push_back(std::vector<int>{spacepointID});
            trackLableToIds[trackLabel] = trkId;
            existTrkIdx++;
        }
    }
    timeInfo.labeling = timer.stopAndGetElapsedTime();
    timeInfo.total = tot_timer.stopAndGetElapsedTime();
}
