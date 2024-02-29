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

Skip::Skip(torch::jit::script::Module module, dc::GameSetting g_game_setting, std::array<std::unique_ptr<dc::ISimulator>, nBatchSize> g_simulators, std::array<std::unique_ptr<dc::IPlayer>, 4> g_players, std::chrono::duration<double> limit,
std::vector<std::vector<double>> win_table,
torch::Device device) : module(module),
g_game_setting(g_game_setting),
g_simulators(std::move(g_simulators)),
g_players(std::move(g_players)),
limit(limit),
win_table(win_table),
device(device),
queue_evaluate()
{}

float Skip::search(std::unique_ptr<UctNode> current_node)
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
        
        current_node->CreateChild(indices.index({0, 0}).item<int>());

        SimulateMove(std::move(current_node), indices.index({0, 0}).item<int>(), child_indices.size());

        // queue child state
        if (queue_evaluate.size() < nBatchSize){
            queue_evaluate.push_back(current_node->GetChild(child_indices.size()));
        };
    } else {
        auto next_node = current_node->GetChild(it - child_indices.begin());
        result = next_node->GetValue();

        // if (result != -1) updateNode(current_node, it - child_indices.begin(), result);

        if (next_node->GetEvaluated() & (next_node->GetGameState().shot > 0)) result = search(std::move(next_node));
    }

    return result;
}


void Skip::updateNode(std::unique_ptr<UctNode> current_node, int child_id, float result){

}



void Skip::SimulateMove(std::unique_ptr<UctNode> current_node, int index, int n_children)
{
    dc::GameState temp_game_state;
    dc::Move temp_moves;
    dc::moves::Shot shots;
    dc::Vector2 velocity;


    temp_game_state = current_node->GetGameState();

    velocity = utility::PixelToVelocity(index % (policy_weight * policy_width) / policy_width, index % (policy_weight * policy_width) % policy_width);

    if (index / (policy_weight * policy_width) == 0) shots = {velocity, dc::moves::Shot::Rotation::kCW};
    else if (index / (policy_weight * policy_width) == 1) shots = {velocity, dc::moves::Shot::Rotation::kCCW};
    else std::cerr << "shot error!";

    auto & current_player = *g_players[temp_game_state.shot / 4];

    dc::ApplyMove(g_game_setting, *g_simulators[0],
        current_player, temp_game_state, temp_moves, std::chrono::milliseconds(0));

    current_node->GetChild(n_children)->SetGameState(temp_game_state);
}


torch::Tensor Skip::EvaluateGameState(std::vector<dc::GameState> game_states, dc::GameSetting game_setting)
{   
    utility::ModelInput model_input = utility::GameStateToInput(game_states, game_setting, device);

    auto outputs = module.forward(model_input.inputs).toTensor().to(torch::kCPU);

    torch::Tensor win_rate = torch::zeros({game_states.size(), game_states[0].kShotPerEnd+1}).to(torch::kCPU);

    for (auto n=0; n < game_states.size(); ++n){    
        for (auto i=0; i < game_states[n].kShotPerEnd+1; ++i){
            int scorediff_after_end = model_input.score.index({n, 0}).item<int>() + i - game_states[n].kShotPerEnd/2;
            if (scorediff_after_end > 9) scorediff_after_end = 9;
            else if (scorediff_after_end < -9) scorediff_after_end = -9;

            win_rate.index({n, i}) = win_table[scorediff_after_end+9][model_input.end.index({n, 0}).item<int>()];
        }
    }

    torch::Tensor win_prob = at::sum(F::softmax(outputs, 1) * win_rate, 1).to(torch::kCPU);

    return win_prob;
}


void Skip::EvaluateQueue()
{
    std::vector<dc::GameState> game_states;
    game_states.resize(queue_evaluate.size());

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
    std::unique_ptr<UctNode> root_node(new UctNode());
    root_node->SetGameState(current_game_state);
    root_node->SetEvaluatedResults(torch::rand({1, policy_weight * policy_width * policy_rotation}), current_outputs.index({0}).item<float>());



    auto now = std::chrono::system_clock::now();    while ((now - start < limit)){
        while (queue_evaluate.size() < nBatchSize) {
            search(std::move(root_node));
        }

        EvaluateQueue();

        // updateNode();
    }

    return dc::moves::Concede();
}

