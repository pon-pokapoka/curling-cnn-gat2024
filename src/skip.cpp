#include "skip.hpp"

#include <iostream>

#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/csrc/api/include/torch/nn/functional/activation.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "utility.hpp"
#include "readcsv.hpp"

namespace F = torch::nn::functional;

// int const policy_weight = 16;
// int const policy_width = 32;
// int const policy_rotation = 2;

Skip::Skip() : module(),
g_game_setting(),
g_simulators(),
g_players(),
limit(),
win_table(),
device(torch::kCPU),
dtype(torch::kBFloat16),
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
    team = g_team;

    win_table = readcsv("model/win_table.csv");

    // for (const auto& row : win_table) {
    //     for (const auto& value : row) {
    //         std::cout << value << ' ';
    //     }
    //     std::cout << std::endl;
    // }

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
        module.to(dtype);
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
        inputs.push_back(torch::rand({nBatchSize, 18, 64, 16}, dtype).to(device));

        // Execute the model and turn its output into a tensor.
        auto outputs = module.forward(inputs).toTensor();
        torch::Tensor out1 = outputs.to(torch::kCPU);
    }
    std::cout << std::endl;
    for (auto i = 0; i < 10; ++i) {
        std::cout << "." << std::flush;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({1, 18, 64, 16}, dtype).to(device));

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

    dc::GameState temp_game_state(g_game_setting);
    kShotPerEnd = static_cast<int>(temp_game_state.kShotPerEnd);

    // 考慮時間制限
    // ショット数で等分するが、超過分を考慮して0.8倍しておく
    limit = g_game_setting.thinking_time[0] * 0.9 / (kShotPerEnd/2) / g_game_setting.max_end;

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

    filt = torch::from_blob(utility::createFilter().data(), {policy_rotation, policy_weight, policy_width}, torch::kBool);

    // std::cout << filt << std::endl;

    // std::cout << utility::createFilter() << std::endl;

    // for (auto i=0; i < policy_rotation; ++i){
    //     for (auto j=0; j < policy_weight; ++j) {
    //         for (auto k=0; k < policy_width; ++k) {
    //             std::cout << filt.index({i, j, k}).item<bool>();
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n\n";
    // }


    // for (auto i=0; i < policy_rotation; ++i){
    //     for (auto j=0; j < policy_weight; ++j) {
    //         for (auto k=0; k < policy_width; ++k) {
    //             std::cout << utility::createFilter()[utility::Id3d1d(i, j, k)];
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n\n";
    // }
}

float Skip::search(UctNode* current_node, int k)
{
    float result = current_node->GetValue();
    // output = evaluate(current_node);
    // set_policy_value(current_node, output);
    // set_filter(current_node)

    // current_node.getPolicy();
    // filt = current_node.getFilter();

    torch::Tensor policy = torch::rand({1, policy_weight*policy_width*policy_rotation}) * filt.reshape({1, policy_weight*policy_width*policy_rotation});
    auto indices = std::get<1>(torch::topk(policy, 1));

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


void Skip::searchById(UctNode* current_node, int k, int index)
{
    auto child_indices = current_node->GetChildIndices();

    // auto it = std::find(child_indices.begin(), child_indices.end(), index);

    // if (it == child_indices.end()) {
        queue_create_child[k] = current_node;
        queue_create_child_index[k] = index;
        flag_create_child[k] = true;
        SimulateMove(current_node, queue_create_child_index[k], k);

    // }

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


std::vector<float> Skip::EvaluateGameState(std::vector<dc::GameState> game_states, dc::GameSetting game_setting)
{
    torch::NoGradGuard no_grad; 
   
    // auto start = std::chrono::system_clock::now();
    // auto now = std::chrono::system_clock::now();

    utility::ModelInput model_input = utility::GameStateToInput(game_states, game_setting, device, dtype);
    // now = std::chrono::system_clock::now();
    // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;




    auto outputs = module.forward(model_input.inputs).toTensor().to(torch::kCPU);
    // now = std::chrono::system_clock::now();
    // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;



    int size = static_cast<int>(game_states.size());
    std::vector<std::vector<float>> win_rate_array(size, std::vector<float>(kShotPerEnd+1));

    // if (size==1) {
    //     auto sheet = model_input.inputs[0].toTensor().to(torch::kCPU);

    //     std::cout << sheet[0][0] << "\n" << sheet[0][1] << "\n";
    //     for (auto i=0; i<16; ++i){
    //         std::cout << sheet[0][i+2] << "\n";
    //     }
    // }
    // if (size!=1) std::cout << F::softmax(outputs, 1)[0] << std::endl;


    torch::Tensor score_prob = F::softmax(outputs, 1);
    std::vector<float> win_prob(size, 0);

    for (auto n=0; n < size; ++n){
        int next_end = 10 - g_game_setting.max_end + model_input.end[n] + 1;
        if (game_states[n].shot == 0) {
            if (game_states[n].IsGameOver()) {
                win_prob[n] = team == game_states[n].game_result->winner;
            } else {
                int scorediff_after_end = model_input.score[n];
                if (scorediff_after_end > 9) scorediff_after_end = 9;
                else if (scorediff_after_end < -9) scorediff_after_end = -9;
                if (team == game_states[n].hammer) 
                win_prob[n] = win_table[scorediff_after_end+9][next_end];
                else win_prob[n] = 1 - win_table[scorediff_after_end+9][next_end];
            }
        } else {
            for (auto i=0; i < kShotPerEnd+1; ++i){
                int scorediff_after_end = model_input.score[n] + i - kShotPerEnd/2;
                if (scorediff_after_end > 9) scorediff_after_end = 9;
                else if (scorediff_after_end < -9) scorediff_after_end = -9;

                if (i > kShotPerEnd/2) {
                    win_rate_array[n][i] = 1 - win_table[9-scorediff_after_end][next_end];
                } else {
                    win_rate_array[n][i] = win_table[scorediff_after_end+9][next_end];
                }
            }

            for (auto i=0; i < kShotPerEnd+1; ++i) {
                win_prob[n] += score_prob.index({n, i}).item<float>() * win_rate_array[n][i];
            }
            if (team != game_states[n].hammer) win_prob[n] = 1 - win_prob[n];
        }
    }
    // now = std::chrono::system_clock::now();
    // std::cout << "Evaluate: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << " msec" << std::endl;

    // torch::Tensor win_rate = torch::from_blob(win_rate_array.data(), {size, kShotPerEnd+1});

    // torch::Tensor win_prob = at::sum(F::softmax(outputs, 1) * win_rate, 1).to(torch::kCPU);
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


    std::vector<float> value = EvaluateGameState(game_states, g_game_setting);


    torch::Tensor policy = torch::rand({size, policy_weight * policy_width * policy_rotation}).to(torch::kCPU);
    // torch::Tensor value = torch::rand({size});

    for (int i=0; i<size; ++i) {
        queue_evaluate[i]->SetEvaluatedResults(policy[i], value[i]);
        // queue_evaluate[i]->SetFilter(utility::createFilter(game_states[i], g_game_setting));
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


    auto sheet = utility::GameStateToInput({current_game_state}, g_game_setting, torch::kCPU, dtype).inputs[0].toTensor();
    // std::cout << sheet[0][0] << sheet[0][1] << std::endl;
    // for (auto i=0; i<16; ++i){
    //     std::cout << sheet[0][i+2][0][0].item<int>();
    // }
    // for (auto i=0; i < 2; ++i){
    //     for (auto j=0; j < utility::height; ++j) {
    //         for (auto k=0; k < utility::width; ++k) {
    //             std::cout << sheet.index({0, i, j, k}).item<float>();
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n";
    //     std::cout << "\n";
    // }

    // for (auto i=0; i < 16; ++i) {
    //     std::cout << sheet[0][i+2][0][0].item<float>();
    // }
    // auto policy = F::softmax(current_outputs.elements()[0].toTensor().reshape({1, 18700}).to(torch::kCPU), 1);

    // auto filt = utility::createFilter(current_game_state, g_game_setting);

    // root node
    std::unique_ptr<UctNode> root_node(new UctNode());
    root_node->SetGameState(current_game_state);
    root_node->SetEvaluatedResults(torch::rand({1, policy_weight * policy_width * policy_rotation}), current_outputs[0]);
    // root_node->SetFilter(filt);

    // std::cout << torch::rand({policy_rotation, policy_weight, policy_width}) * filt << std::endl;

    int count = 0;
    auto now = std::chrono::system_clock::now();
    // auto a = std::chrono::system_clock::now();
    // auto b = std::chrono::system_clock::now();
    while ((now - start < limit)){
        // a = std::chrono::system_clock::now();
        if (current_game_state.shot + 1 == kShotPerEnd) {
            if (count >= policy_rotation*policy_weight*policy_width*nSimulation) break;
            #pragma omp parallel for
            for (auto i = 0; i < nBatchSize; ++i) {
                searchById(root_node.get(), i, (count + i) % (policy_rotation*policy_weight*policy_width));
            }

        } else {
            // while (queue_evaluate.size() < nBatchSize) {
            #pragma omp parallel for
            for (auto i = 0; i < nLoop; ++i) {
                // std::cout << i << "  ";
                search(root_node.get(), i);
                if (std::accumulate(std::begin(flag_create_child), std::end(flag_create_child), 0) >= nBatchSize) continue;
            }
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


        now = std::chrono::system_clock::now();
    }
    // GPUのキャッシュをクリア
    c10::cuda::CUDACachingAllocator::emptyCache();

    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << count << " simulations in " << msec << " msec" << std::endl;

    std::vector<int> child_indices = root_node->GetChildIndices();
    std::vector<float> values;
    std::array<std::array<std::array<float, policy_width>, policy_weight>, policy_rotation> win_rate{{}};
    if (current_game_state.shot + 1 == kShotPerEnd) {
        values.resize(policy_width*policy_weight*policy_rotation, 0);
        for (auto i=0; i < child_indices.size(); ++i) {
            values[child_indices[i]] += root_node->GetChildById(child_indices[i])->GetCountValue() / nSimulation;
        }
        // for (auto i=0; i < policy_rotation; ++i){
        //     for (auto j=0; j < policy_weight; ++j) {
        //         for (auto k=0; k < policy_width; ++k) {
        //             win_rate[i][j][k] = values[utility::Id3d1d(i, j, k)];
        //         }
        //     }
        // }
    } else {
        for (auto index: child_indices) {
            values.push_back(root_node->GetChild(index)->GetCountValue());
        }
        // for (auto index: child_indices) {
        //     win_rate[index / (policy_weight * policy_width)][index % (policy_weight * policy_width) / policy_width][index % (policy_weight * policy_width) % policy_width] = root_node->GetChild(index)->GetCountValue();
        // }

    }


    // for (auto i=0; i < policy_rotation; ++i){
    //     // for (auto j=0; j < policy_weight; ++j) {
    //         int j = 0;
    //         for (auto k=0; k < policy_width; ++k) {
    //             std::cout << std::setw(2) << std::setfill('0') << std::setprecision(0) << static_cast<int>(win_rate[i][j][k] * 100) << " ";
    //         }
    //         std::cout << "\n";
    //     // }
    //     std::cout << "\n";
    //     std::cout << "\n";
    // }



    int pixel_id = child_indices[static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())))];

    // std::cout << pixel_id << ":   " << pixel_id / (policy_weight * policy_width) << ", " << pixel_id % (policy_weight * policy_width) / policy_width << ", " << pixel_id % (policy_weight * policy_width) % policy_width << std::endl;

    dc::moves::Shot::Rotation rotation;
    if (pixel_id / (policy_weight * policy_width) == 0) rotation = dc::moves::Shot::Rotation::kCW;
    else rotation = dc::moves::Shot::Rotation::kCCW;

    dc::moves::Shot shot = {utility::PixelToVelocity(pixel_id % (policy_weight * policy_width) / policy_width, pixel_id % (policy_weight * policy_width) % policy_width), rotation};
    dc::Move move = shot;

    // std::cout << std::accumulate(std::begin(values), std::end(values), 0.f) << std::endl;
    if (std::accumulate(std::begin(values), std::end(values), 0.f) < 1e-6f) move = dc::moves::Concede();

    // move = dc::moves::Concede();

    return move;
}

