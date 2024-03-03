#include "skip.hpp"

#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/csrc/api/include/torch/nn/functional/activation.h>

#include "utility.hpp"

namespace F = torch::nn::functional;

int height = 64;
int width = 16;
int nChannel = 18;

int policy_weight = 16;
int policy_width = 32;
int policy_rotation = 2;

Skip::Skip(torch::jit::script::Module module, dc::GameSetting g_game_setting, std::array<std::shared_ptr<dc::ISimulator>, nLoop> g_simulators, std::array<std::shared_ptr<dc::IPlayer>, 4> g_players, std::chrono::duration<double> limit,
std::vector<std::vector<double>> win_table,
torch::Device device) : module(module),
g_game_setting(g_game_setting),
g_simulators(g_simulators),
g_players(g_players),
limit(limit),
win_table(win_table),
device(device),
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

float Skip::search(std::shared_ptr<UctNode> current_node, int k)
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

        // if (result != -1) updateNode(current_node, it - child_indices.begin(), result);

        if (next_node->GetEvaluated() & (next_node->GetGameState().shot > 0)) result = search(next_node, k);
    }

    return result;
}


void Skip::updateNode(std::shared_ptr<UctNode> current_node, int child_id, float result){

}



void Skip::SimulateMove(std::shared_ptr<UctNode> current_node, int index, int k)
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
    auto start = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::now();

    utility::ModelInput model_input = utility::GameStateToInput(game_states, game_setting, torch::kCPU);
    now = std::chrono::system_clock::now();
    std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;




    auto outputs = module.forward(model_input.to(device).inputs).toTensor().to(torch::kCPU);
    now = std::chrono::system_clock::now();
    std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;


    torch::Tensor win_rate = torch::zeros({game_states.size(), game_states[0].kShotPerEnd+1}).to(torch::kCPU);

    for (auto n=0; n < game_states.size(); ++n){    
        for (auto i=0; i < game_states[n].kShotPerEnd+1; ++i){
            int scorediff_after_end = model_input.score[n] + i - game_states[n].kShotPerEnd/2;
            if (scorediff_after_end > 9) scorediff_after_end = 9;
            else if (scorediff_after_end < -9) scorediff_after_end = -9;

            win_rate.index({n, i}) = win_table[scorediff_after_end+9][model_input.end[n]];
        }
    }
    now = std::chrono::system_clock::now();
    std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;


    torch::Tensor win_prob = at::sum(F::softmax(outputs, 1) * win_rate, 1).to(torch::kCPU);
    now = std::chrono::system_clock::now();
    std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;


    return win_prob;
}


void Skip::EvaluateQueue()
{       
  
    std::vector<dc::GameState> game_states;
    game_states.resize(queue_evaluate.size());

    #pragma omp parallel for
    for (int i=0; i<queue_evaluate.size(); ++i) {
        game_states[i] = queue_evaluate[i]->GetGameState();
    }


    torch::Tensor value = EvaluateGameState(game_states, g_game_setting).to(torch::kCPU);


    torch::Tensor policy = torch::rand({queue_evaluate.size(), policy_weight * policy_width * policy_rotation}).to(torch::kCPU);
    // torch::Tensor value = torch::rand({queue_evaluate.size()});

    for (int i=0; i<queue_evaluate.size(); ++i) {
        queue_evaluate[i]->SetEvaluatedResults(policy[i], value[i].item<float>());
    }
    queue_evaluate.clear();
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
    std::shared_ptr<UctNode> root_node(new UctNode());
    root_node->SetGameState(current_game_state);
    root_node->SetEvaluatedResults(torch::rand({1, policy_weight * policy_width * policy_rotation}), current_outputs.index({0}).item<float>());

    int count = 0;
    auto now = std::chrono::system_clock::now();
    while ((now - start < limit)){
        // while (queue_evaluate.size() < nBatchSize) {
        #pragma omp parallel for
        for (auto i = 0; i < nLoop; ++i) {
            // std::cout << i << "  ";
            search(root_node, i);
            if (std::accumulate(std::begin(flag_create_child), std::end(flag_create_child), 0) >= nBatchSize) continue;
        }
        now = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;
        
        for (auto i = 0; i < nLoop; ++i) {
            if (flag_create_child[i]) {
                queue_create_child[i]->CreateChild(queue_create_child_index[i]);
                queue_create_child[i]->GetChild(queue_create_child_index[i])->SetGameState(temp_game_states[i]);
                if (queue_evaluate.size() < nBatchSize) {
                    queue_evaluate.push_back(queue_create_child[i]->GetChild(queue_create_child_index[i]));
                }
            }
        }

        for (auto& ptr : queue_create_child) {
            ptr.reset();
        }
        #pragma omp parallel for
        for (auto i=0; i < nLoop; ++i) {
            flag_create_child[i] = false;
        }


        count += queue_evaluate.size();
        EvaluateQueue();

        // updateNode();
        now = std::chrono::system_clock::now();
    }

    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << count << " simulations in " << msec << " msec" << std::endl;
    dc::moves::Shot shot = {utility::PixelToVelocity(8, 8), dc::moves::Shot::Rotation::kCCW};
    dc::Move move = shot;
    return move;
}

