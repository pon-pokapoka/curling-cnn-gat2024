#include "uctnode.hpp"

int height = 64;
int width = 16;
int nChannel = 18;

int policy_weight = 16;
int policy_width = 32;
int policy_rotation = 2;

void search(UctNode* current_node, std::vector<UctNode*> queue_evaluate)
{
    // output = evaluate(current_node);
    // set_policy_value(current_node, output);
    // set_filter(current_node)

    // current_node.getPolicy();
    // filt = current_node.getFilter();

    auto indices = std::get<1>(torch::topk(torch::rand({1, policy_weight*policy_width*policy_rotation}) * filt.reshape({1, policy_weight*policy_width*policy_rotation}), 1));

    child_indices = current_node->GetChildIndices();

    auto it = std::find(child_indices.begin(), child_indices.end(), indices.index({0, 0}).item<int>());

    if (it == child_indices.end()) {
        
        current_node->CreateChild(indices.index({0, 0}).item<int>());

        SimulateMove(current_node, indices.index({0, 0}).item<int>(), child_indices.size());

        // queue child state
        queue_evaluate.push_back(current_node->GetChild(child_indices.size()));
    } else {
        auto next_node = current_node->getChild(it - child_indices.begin());

        if (next_node->GetEvaluated()) search(next_node, queue_evaluate);
    }
}


class ModelInput {
    public:
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor end;
        torch::Tensor score;
};

// GameStateからモデルに入力する形式に変換する
ModelInput GameStateToInput(std::vector<dc::GameState> game_states, dc::GameSetting game_setting)
{
    std::vector<torch::jit::IValue> inputs;

    torch::Tensor sheet = torch::zeros({static_cast<int>(game_states.size()), nChannel, height, width}).to(device);
    torch::Tensor end = torch::zeros({static_cast<int>(game_states.size()), 1}).to(device);
    torch::Tensor score = torch::zeros({static_cast<int>(game_states.size()), 1}).to(device);
    // torch::Tensor shot = torch::zeros({static_cast<int>(game_states.size()), 1}).to(device);

    for (size_t k=0; k < game_states.size(); ++k){
        int i = static_cast<int>(k);
        if (game_states[i].IsGameOver()) continue; // 試合終了していたらスキップ

        // shot.index({i, 0}) = (game_states[i].kShotPerEnd - game_states[i].shot) / 16.f;
        score.index({i, 0}) = (static_cast<float>(game_states[i].GetTotalScore(game_states[i].hammer)) - static_cast<float>(game_states[i].GetTotalScore(dc::GetOpponentTeam(game_states[i].hammer))));
        if (game_states[i].end < game_setting.max_end){
            end.index({i, 0}) = game_states[i].end;
        } else {
            end.index({i, 0}) = 10; // エキストラエンドは10エンドと同じ
        }
        // std::cout << game_states[i].kShotPerEnd << std::endl;
        
        for (auto l=0; l <= game_states[i].shot; ++l){
            for (auto n=0; n < height; ++n){
                for (auto m=0; m < width; ++m){
                    sheet.index({i, l+2, n, m}) = 1;
                }
            }  
        }

        for (size_t team_stone_idx = 0; team_stone_idx < game_states[i].kShotPerEnd / 2; ++team_stone_idx) {
            auto const& stone = game_states[i].stones[static_cast<size_t>(dc::GetOpponentTeam(game_states[i].hammer))][team_stone_idx];
            // std::cout << stone_nohammer->position.x << std::endl;
            if (stone) {
                std::pair <int, int> pixel = PositionToPixel(stone->position);
                // std::cout << pixel << std::endl;
                sheet.index({i, 0, pixel.first, pixel.second}) = 1;
            }
        }
        for (size_t team_stone_idx = 0; team_stone_idx < game_states[i].kShotPerEnd / 2; ++team_stone_idx) {
            auto const& stone = game_states[i].stones[static_cast<size_t>(game_states[i].hammer)][team_stone_idx];
            if (stone) {
                std::pair <int, int> pixel = PositionToPixel(stone->position);
                // std::cout << pixel << std::endl;
                sheet.index({i, 1, pixel.first, pixel.second}) = 1;
            }
        }
    }

    inputs.push_back(sheet);
    // inputs.push_back(end);
    // inputs.push_back(score);
    // inputs.push_back(shot);

    ModelInput model_input;

    model_input.inputs = inputs;
    model_input.end = end;
    model_input.score = score;

    return model_input;
}



void SimulateMove(UctNode* current_node, int index, int n_children)
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


void EvaluateGameState(dc::GameState game_states, dc::GameSetting game_setting)
{   
    ModelInput model_input = GameStateToInput(game_states, game_setting);

    auto outputs = module.forward(model_input.inputs).toTuple();

    for (auto n=0; n < game_states.size(); ++n){    
        for (auto i=0; i < kShotPerEnd+1; ++i){
            int scorediff_after_end = model_input.score({n, 0}) + i - kShotPerEnd/2;
            if (scorediff_after_end > 9) scorediff_after_end = 9;
            else if (scorediff_after_end < -9) scorediff_after_end = -9;

            win_rate({n, i}) = win_table[scorediff_after_end+9, model_input.end({n, 0})];
        }
    }

    Torch::Tensor win_prob = at::sum(F::softmax(outputs->elements()[0].toTensor(), 1) * win_rate, 1);

    return win_prob;
}


void EvaluateQueue(std::vector<UctNode*> queue)
{
    std::vector<dc::GameState> game_states;
    game_states.resize(queue.size());

    for (int i=0; i<queue.size(); ++i) {
        game_states[i] = queue[i]->GetGameState();
    }

    Torch::Tensor value = EvaluateGameState(game_states, game_setting);

    for (int i=0; i<queue.size(); ++i) {
        policy[i] = torch::rand({1, policy_weight * policy_width * policy_rotation});
    }
    // Torch::Tensor value = torch::rand({queue.size()});

    for (int i=0; i<queue.size(); ++i) {
        queue->SetEvaluatedResults(policy[i], value[i]);
    }

}


void OnMyTurn(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
    auto start = std::chrono::system_clock::now();

    dc::GameState current_game_state = game_state;

    std::unique_ptr<UctNode> root_node(new UctNode());
    root_node->SetGameState(current_game_state);

    torch::NoGradGuard no_grad; 

    // 現在の局面を評価
    auto current_outputs = module.forward(GameStateToInput({current_game_state}, g_game_setting)).toTuple();

    auto policy = F::softmax(current_outputs->elements()[0].toTensor().reshape({1, 18700}).to(torch::kCPU), 1);

    torch::Tensor filt = createFilter(current_game_state, g_game_setting);

    // root node
    std::unique_ptr root_node(new UctNode());
    root_node.SetGameState(current_game_state);

    std::vector<UctNode*> queue_evaluate;

    for (int i=0; i<batch_size; ++i) {
        search(root_node, queue_evaluate);
    }

    EvaluateQueue();
}


// ストーンの座標からシートの画像のピクセルに変換する
std::pair<int, int> PositionToPixel(dc::Vector2 position)
{
    std::pair<int, int> pixel;

    pixel.first = static_cast<int>(round((position.y - 32.004)/0.0254/16));
    pixel.second = 8 - static_cast<int>(round(position.x/0.0254/16));

    return pixel;
}


std::vector<torch::jit::IValue> GameStateToInput(std::vector<dc::GameState> game_states, dc::GameSetting game_setting)
{
    std::vector<torch::jit::IValue> inputs;

    torch::Tensor sheet = torch::zeros({static_cast<int>(game_states.size()), 18, 64, 16}).to(device);

    for (size_t k=0; k < game_states.size(); ++k){
        int i = static_cast<int>(k);
        if (game_states[i].IsGameOver()) continue; // 試合終了していたらスキップ

        for (auto j=0; j<game_states[i].shot+1; ++j) {
            for (auto m=0; m < 64; ++m){
                for (auto n=0; n < 16; ++n){
                    sheet.index({i, j+2, m, n}) = 1;
                }
            }
        }
        // score.index({i, 0}) = (static_cast<float>(game_states[i].GetTotalScore(game_states[i].hammer)) - static_cast<float>(game_states[i].GetTotalScore(dc::GetOpponentTeam(game_states[i].hammer)))) * 0.1f + 0.5f;
        // if (game_states[i].end < game_setting.max_end){
        //     end.index({i, 0}) = (game_setting.max_end - game_states[i].end) / 10.f;
        // } else {
        //     end.index({i, 0}) = 0.1f; // エキストラエンドは10エンドと同じ
        // }
        // std::cout << game_states[i].kShotPerEnd << std::endl;
        for (size_t team_stone_idx = 0; team_stone_idx < game_states[i].kShotPerEnd / 2; ++team_stone_idx) {
            auto const& stone = game_states[i].stones[static_cast<size_t>(dc::GetOpponentTeam(game_states[i].hammer))][team_stone_idx];
            // std::cout << stone_nohammer->position.x << std::endl;
            if (stone) {
                std::pair <int, int> pixel = PositionToPixel(stone->position);
                // std::cout << pixel << std::endl;
                sheet.index({i, 0, pixel.first, pixel.second}) = 1;
            }
        }
        for (size_t team_stone_idx = 0; team_stone_idx < game_states[i].kShotPerEnd / 2; ++team_stone_idx) {
            auto const& stone = game_states[i].stones[static_cast<size_t>(game_states[i].hammer)][team_stone_idx];
            if (stone) {
                std::pair <int, int> pixel = PositionToPixel(stone->position);
                // std::cout << pixel << std::endl;
                sheet.index({i, 1, pixel.first, pixel.second}) = 1;
            }
        }
    }

    inputs.push_back(sheet);
    // inputs.push_back(end);
    // inputs.push_back(score);
    // inputs.push_back(shot);

    return inputs;
}
