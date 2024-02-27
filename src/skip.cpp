#include "skip.hpp"

#include <torch/script.h>
#include <torch/cuda.h>

namespace F = torch::nn::functional;

int height = 64;
int width = 16;
int nChannel = 18;

int policy_weight = 16;
int policy_width = 32;
int policy_rotation = 2;

Skip::Skip(torch::jit::script::Module module, dc::GameSetting g_game_setting, std::vector<std::unique_ptr<dc::ISimulator>> g_simulators) : module(module),
g_game_setting(g_game_setting),
g_simulators(g_simulators)
{}

float Skip::search(std::unique_ptr<UctNode> current_node, std::vector<std::unique_ptr<UctNode>> queue_evaluate)
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

        SimulateMove(current_node, indices.index({0, 0}).item<int>(), child_indices.size());

        // queue child state
        if (queue_evaluate.size() < nBatchSize){
            queue_evaluate.push_back(current_node->GetChild(child_indices.size()));
        };
    } else {
        auto next_node = current_node->GetChild(it - child_indices.begin());
        result = next_node->GetValue();

        // if (result != -1) updateNode(current_node, it - child_indices.begin(), result);

        if (next_node->GetEvaluated() & (next_node->GetGameState().shot > 0)) result = search(next_node, queue_evaluate);
    }

    return result;
}


void Skip::updateNode(std::unique_ptr<UctNode> current_node, int child_id, float result){

}



void Skip::SimulateMove(std::unique_ptr<UctNode> current_node, int index, int n_children)
{
    temp_game_states[i] = current_node->GetGameState();

    velocity[i] = PixelToVelocity(index % (policy_weight * policy_width) / policy_width, index % (policy_weight * policy_width) % policy_width);

    if (index / (policy_weight * policy_width) == 0) shots[i] = {velocity[i], dc::moves::Shot::Rotation::kCW};
    else if (index / (policy_weight * policy_width) == 1) shots[i] = {velocity[i], dc::moves::Shot::Rotation::kCCW};
    else std::cerr << "shot error!";


    dc::ApplyMove(g_game_setting, *g_simulators[i],
        current_player, temp_game_states[i], temp_moves[i], std::chrono::milliseconds(0));

    current_node->GetChild(n_children)->SetGameState(temp_game_states[i]);
}


void Skip::EvaluateGameState(std::vector<dc::GameState> game_states, dc::GameSetting game_setting)
{   
    utility::ModelInput model_input = utility::GameStateToInput(game_states, game_setting);

    auto outputs = module.forward(model_input.inputs).toTuple();

    for (auto n=0; n < game_states.size(); ++n){    
        for (auto i=0; i < kShotPerEnd+1; ++i){
            int scorediff_after_end = model_input.score({n, 0}) + i - kShotPerEnd/2;
            if (scorediff_after_end > 9) scorediff_after_end = 9;
            else if (scorediff_after_end < -9) scorediff_after_end = -9;

            win_rate({n, i}) = win_table[scorediff_after_end+9, model_input.end({n, 0})];
        }
    }

    torch::Tensor win_prob = at::sum(F::softmax(outputs->elements()[0].toTensor(), 1) * win_rate, 1);

    return win_prob;
}


void Skip::EvaluateQueue(std::vector<std::unique_ptr<UctNode>> queue)
{
    std::vector<dc::GameState> game_states;
    game_states.resize(queue.size());

    for (int i=0; i<queue.size(); ++i) {
        game_states[i] = queue[i]->GetGameState();
    }

    torch::Tensor value = EvaluateGameState(game_states, g_game_setting);

    for (int i=0; i<queue.size(); ++i) {
        policy[i] = torch::rand({1, policy_weight * policy_width * policy_rotation});
    }
    // torch::Tensor value = torch::rand({queue.size()});

    for (int i=0; i<queue.size(); ++i) {
        queue[i]->SetEvaluatedResults(policy[i], value[i]);
    }

    queue.clear()
}


void Skip::command(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
    auto start = std::chrono::system_clock::now();

    dc::GameState current_game_state = game_state;

    std::unique_ptr<UctNode> root_node(new UctNode());
    root_node->SetGameState(current_game_state);

    torch::NoGradGuard no_grad; 

    // 現在の局面を評価
    auto current_outputs = EvaluateGameState(current_game_state, game_setting);

    auto policy = F::softmax(current_outputs->elements()[0].toTensor().reshape({1, 18700}).to(torch::kCPU), 1);

    // torch::Tensor filt = createFilter(current_game_state, g_game_setting);

    // root node
    std::unique_ptr<UctNode> root_node(new UctNode());
    root_node->SetGameState(current_game_state);
    root_node->SetEvaluatedResults(torch::rand({1, policy_weight * policy_width * policy_rotation}), current_outputs[0]);

    std::vector<std::unique_ptr<UctNode>> queue_evaluate;

    while ((now - start < limit)){
        while (queue_evaluate.size() < nBatchSize) {
            search(root_node, queue_evaluate);
        }

        EvaluateQueue();

        // updateNode();
    }

    return dc::moves::Concede();
}

