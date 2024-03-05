#include "skip.hpp"

#include <iostream>

#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/csrc/api/include/torch/nn/functional/activation.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "utility.hpp"
#include "readcsv.hpp"

namespace F = torch::nn::functional;

int height = 64;
int width = 16;
int nChannel = 18;

int policy_weight = 16;
int policy_width = 32;
int policy_rotation = 2;

Skip::Skip() : module(),
g_game_setting(),
g_simulators(),
g_players(),
limit(),
win_table(),
device(torch::kCPU),
queue_evaluate(),
queue_simulate(),
queue_create_child(),
queue_create_child_index(),
flag_create_child(),
temp_game_states()
{
    #pragma omp parallel for
    for (auto i=0; i < nLoop; ++i) {
        flag_create_child[i] = false;
    }
}

void Skip::OnInit(dc::Team const g_team, dc::GameSetting const& game_setting, std::unique_ptr<dc::ISimulatorFactory> simulator_factory,     std::array<std::unique_ptr<dc::IPlayerFactory>, 4> player_factories,  std::array<size_t, 4> & player_order)
{
    win_table = readcsv("model/win_table.csv");

    for (const auto& row : win_table) {
        for (const auto& value : row) {
            std::cout << value << '\t';
        }
        std::cout << std::endl;
    }

    torch::NoGradGuard no_grad; 

    device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        device = torch::kCUDA;
    }   
    else {
        std::cout << "CUDA is not available." << std::endl;
    }
    // Deserialize the ScriptModule from a file using torch::jit::load().
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout << "model loading..." << std::endl;
        module = torch::jit::load("model/traced_curling_shotpyshot_v2_score-008.pt", device);
        module.to(torch::kBFloat16);
        std::cout << "model loaded" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    // ここでCNNによる推論を行うことで、次回以降の速度が早くなる
    // 使うバッチサイズすべてで行っておく
    std::cout << "initial inference\n";
    for (auto i = 0; i < 10; ++i) {
        std::cout << "." << std::flush;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({nBatchSize, 18, 64, 16}, torch::kBFloat16).to(device));

        // Execute the model and turn its output into a tensor.
        auto outputs = module.forward(inputs).toTensor();
        torch::Tensor out1 = outputs.to(torch::kCPU);
    }
    std::cout << std::endl;
    for (auto i = 0; i < 10; ++i) {
        std::cout << "." << std::flush;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({1, 18, 64, 16}, torch::kBFloat16).to(device));

        // Execute the model and turn its output into a tensor.
        auto outputs = module.forward(inputs).toTensor();
        torch::Tensor out1 = outputs.to(torch::kCPU);
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::cout << std::endl;

    // シミュレータFCV1Lightを使用する．
    g_game_setting = game_setting;
    for (unsigned i = 0; i < nLoop; ++i) {
        g_simulators[i] = dc::simulators::SimulatorFCV1LightFactory().CreateSimulator();
    }
    g_simulator_storage = g_simulators[0]->CreateStorage();

    // プレイヤーを生成する
    // 非対応の場合は NormalDistプレイヤーを使用する．
    assert(g_players.size() == player_factories.size());
    for (size_t i = 0; i < g_players.size(); ++i) {
        auto const& player_factory = player_factories[player_order[i]];
        if (player_factory) {
            g_players[i] = player_factory->CreatePlayer();
        } else {
            g_players[i] = dc::players::PlayerNormalDistFactory().CreatePlayer();
        }
    }

    // 考慮時間制限
    // ショット数で等分するが、超過分を考慮して0.8倍しておく
    limit = g_game_setting.thinking_time[0] * 0.8 / 8. / g_game_setting.max_end;

    dc::GameState temp_game_state(g_game_setting);
    kShotPerEnd = static_cast<int>(temp_game_state.kShotPerEnd);

    // ショットシミュレーションの動作確認
    // しなくて良い
    std::cout << "initial simulation\n";
    for (auto j = 0; j < 10; ++j) {
        std::cout << "." << std::flush;
        dc::GameState dummy_game_state(g_game_setting);
        std::array<dc::GameState, nBatchSize> dummy_game_states; 
        std::array<dc::Move, nBatchSize> dummy_moves;
        auto & dummy_player = *g_players[0];
        dc::moves::Shot dummy_shot = {dc::Vector2(0, 2.5), dc::moves::Shot::Rotation::kCW};
        #pragma omp parallel for
        for (auto i=0; i < nBatchSize; ++i) {
            dummy_game_states[i] = dummy_game_state;
        }
        #pragma omp parallel for
        for (auto i=0; i < nBatchSize; ++i) {
            dummy_moves[i] = dummy_shot;
            g_simulators[i]->Load(*g_simulator_storage);

            dc::ApplyMove(g_game_setting, *g_simulators[i],
                dummy_player, dummy_game_states[i], dummy_moves[i], std::chrono::milliseconds(0));
        }
    }
    std::cout << std::endl;


}

float Skip::search(UctNode* current_node, int k)
{
    float result = current_node->GetValue();
    // output = evaluate(current_node);
    // set_policy_value(current_node, output);
    // set_filter(current_node)

    // current_node.getPolicy();
    // filt = current_node.getFilter();

    auto indices = std::get<1>(torch::topk(torch::rand({1, policy_weight*policy_width*policy_rotation}), 1));

    auto child_indices = current_node->GetChildIndices();

    auto it = std::find(child_indices.begin(), child_indices.end(), indices.index({0, 0}).item<int>());

    if (it == child_indices.end()) {
        queue_create_child[k] = current_node;
        queue_create_child_index[k] = indices.index({0, 0}).item<int>();
        flag_create_child[k] = true;
        SimulateMove(current_node, queue_create_child_index[k], k);

    } else {
        auto next_node = current_node->GetChild(indices.index({0, 0}).item<int>());
        if (next_node->GetEvaluated()) result = next_node->GetValue();
        else {
            queue_create_child[k] = current_node;
            queue_create_child_index[k] = indices.index({0, 0}).item<int>();
            flag_create_child[k] = true;
        }

        // if (result != -1) updateNode(current_node, it - child_indices.begin(), result);

        if (next_node->GetEvaluated() & (next_node->GetGameState().shot > 0)) result = search(next_node, k);
    }

    return result;
}


void Skip::updateParent(UctNode* node, float value)
{
    if (node->GetParent()) {
        node->GetParent()->SetCountValue(value);
        updateParent(node->GetParent(), value);
    }
}

void Skip::updateNodes()
{
    for (auto& node: queue_evaluate) {
        float value = node->GetValue();
        node->SetCountValue(value);
        updateParent(node, value);
    }
}



void Skip::SimulateMove(UctNode* current_node, int index, int k)
{
    dc::Move temp_move;
    dc::moves::Shot shot;
    dc::Vector2 velocity;


    temp_game_states[k] = current_node->GetGameState();

    velocity = utility::PixelToVelocity(index % (policy_weight * policy_width) / policy_width, index % (policy_weight * policy_width) % policy_width);

    if (index / (policy_weight * policy_width) == 0) shot = {velocity, dc::moves::Shot::Rotation::kCW};
    else if (index / (policy_weight * policy_width) == 1) shot = {velocity, dc::moves::Shot::Rotation::kCCW};
    else std::cerr << "shot error!";

    auto & current_player = *g_players[temp_game_states[k].shot / 4];
    temp_move = shot;

    dc::ApplyMove(g_game_setting, *g_simulators[k],
        current_player, temp_game_states[k], temp_move, std::chrono::milliseconds(0));

}


torch::Tensor Skip::EvaluateGameState(std::vector<dc::GameState> game_states, dc::GameSetting game_setting)
{
    torch::NoGradGuard no_grad; 
   
    // auto start = std::chrono::system_clock::now();
    // auto now = std::chrono::system_clock::now();

    utility::ModelInput model_input = utility::GameStateToInput(game_states, game_setting, device);
    // now = std::chrono::system_clock::now();
    // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;




    auto outputs = module.forward(model_input.inputs).toTensor().to(torch::kCPU);
    // now = std::chrono::system_clock::now();
    // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;


    int size = static_cast<int>(game_states.size());
    torch::Tensor win_rate = torch::zeros({size, kShotPerEnd+1}).to(torch::kCPU);

    for (auto n=0; n < size; ++n){    
        for (auto i=0; i < kShotPerEnd+1; ++i){
            int scorediff_after_end = model_input.score[n] + i - kShotPerEnd/2;
            if (scorediff_after_end > 9) scorediff_after_end = 9;
            else if (scorediff_after_end < -9) scorediff_after_end = -9;

            win_rate.index({n, i}) = win_table[scorediff_after_end+9][model_input.end[n]];
        }
    }
    // now = std::chrono::system_clock::now();
    // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;


    torch::Tensor win_prob = at::sum(F::softmax(outputs, 1) * win_rate, 1).to(torch::kCPU);
    // now = std::chrono::system_clock::now();
    // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;


    return win_prob;
}


void Skip::EvaluateQueue()
{       
    int size = static_cast<int>(queue_evaluate.size());
    std::vector<dc::GameState> game_states;
    game_states.resize(size);

    #pragma omp parallel for
    for (auto i=0; i<size; ++i) {
        game_states[i] = queue_evaluate[i]->GetGameState();
    }


    torch::Tensor value = EvaluateGameState(game_states, g_game_setting).to(torch::kCPU);


    torch::Tensor policy = torch::rand({size, policy_weight * policy_width * policy_rotation}).to(torch::kCPU);
    // torch::Tensor value = torch::rand({size});

    for (int i=0; i<size; ++i) {
        queue_evaluate[i]->SetEvaluatedResults(policy[i], value[i].item<float>());
    }
}


dc::Move Skip::command(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
    auto start = std::chrono::system_clock::now();

    dc::GameState current_game_state = game_state;

    torch::NoGradGuard no_grad; 

    // 現在の局面を評価
    auto current_outputs = EvaluateGameState({current_game_state}, g_game_setting);

    // auto policy = F::softmax(current_outputs.elements()[0].toTensor().reshape({1, 18700}).to(torch::kCPU), 1);

    // torch::Tensor filt = createFilter(current_game_state, g_game_setting);

    // root node
    std::unique_ptr<UctNode> root_node(new UctNode());
    root_node->SetGameState(current_game_state);
    root_node->SetEvaluatedResults(torch::rand({1, policy_weight * policy_width * policy_rotation}), current_outputs.index({0}).item<float>());

    int count = 0;
    auto now = std::chrono::system_clock::now();
    // auto a = std::chrono::system_clock::now();
    // auto b = std::chrono::system_clock::now();
    while ((now - start < limit)){
        // a = std::chrono::system_clock::now();
        // while (queue_evaluate.size() < nBatchSize) {
        #pragma omp parallel for
        for (auto i = 0; i < nLoop; ++i) {
            // std::cout << i << "  ";
            search(root_node.get(), i);
            if (std::accumulate(std::begin(flag_create_child), std::end(flag_create_child), 0) >= nBatchSize) continue;
        }
        // b = std::chrono::system_clock::now();
        // std::cout << "Search: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " msec" << "\n";
        

        // a = std::chrono::system_clock::now();
        for (auto i = 0; i < nLoop; ++i) {
            if (flag_create_child[i]) {
                queue_create_child[i]->CreateChild(queue_create_child_index[i]);
                queue_create_child[i]->GetChild(queue_create_child_index[i])->SetGameState(temp_game_states[i]);
                if (queue_evaluate.size() < nBatchSize) {
                    queue_evaluate.push_back(queue_create_child[i]->GetChild(queue_create_child_index[i]));
                }
            }
        }
        // b = std::chrono::system_clock::now();
        // std::cout << "Expand: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " msec" << "\n";

        // for (auto ptr : queue_create_child) {
        //     ptr.reset();
        // }
        #pragma omp parallel for
        for (auto i=0; i < nLoop; ++i) {
            flag_create_child[i] = false;
        }

        // a = std::chrono::system_clock::now();

        count += queue_evaluate.size();
        EvaluateQueue();
        // b = std::chrono::system_clock::now();
        // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " msec" << "\n";

        // a = std::chrono::system_clock::now();

        updateNodes();
        queue_evaluate.clear();
        // b = std::chrono::system_clock::now();
        // std::cout << "Update: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " msec" << "\n";


        // updateNode();
        now = std::chrono::system_clock::now();
    }
    // GPUのキャッシュをクリア
    c10::cuda::CUDACachingAllocator::emptyCache();

    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << count << " simulations in " << msec << " msec" << std::endl;

    std::vector<int> child_indices = root_node->GetChildIndices();
    std::vector<float> values;
    for (auto index: child_indices) {
        values.push_back(root_node->GetChild(index)->GetValue());
    }

    int pixel_id = child_indices[static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())))];

    dc::moves::Shot shot = {utility::PixelToVelocity(pixel_id % (policy_weight * policy_width) / policy_width, pixel_id % (policy_weight * policy_width) % policy_width), dc::moves::Shot::Rotation::kCCW};
    dc::Move move = shot;
    return move;
}

