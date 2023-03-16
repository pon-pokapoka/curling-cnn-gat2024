#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include "digitalcurling3/digitalcurling3.hpp"

#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/csrc/api/include/torch/nn/functional/activation.h>
#include <c10/cuda/CUDACachingAllocator.h>


namespace dc = digitalcurling3;
namespace F = torch::nn::functional;

const int nSimulation = 4;
const int nBatchSize = 100;
const int nCandidate = 2000;


namespace {


dc::Team g_team;  // 自身のチームID
torch::jit::script::Module module;
dc::GameSetting g_game_setting;
std::unique_ptr<dc::ISimulator> g_simulator;
std::array<std::unique_ptr<dc::ISimulator>, nBatchSize> g_simulators;
std::unique_ptr<dc::ISimulatorStorage> g_simulator_storage;
std::array<std::unique_ptr<dc::IPlayer>, 4> g_players;

std::chrono::duration<double> limit;

torch::Device device(torch::kCPU);


std::pair<int, int> PositionToPixel(dc::Vector2 position)
{
    std::pair<int, int> pixel;

    pixel.first = static_cast<int>(round((position.y - 32.004)/0.0254));
    pixel.second = 93 - static_cast<int>(round(position.x/0.0254));

    return pixel;
}


dc::Vector2 PixelToVelocity(int i, int j)
{
    std::array<float, 50> velocity_array{{2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29, 2.3 ,
        2.31, 2.32, 2.33, 2.34 , 2.345, 2.35 , 2.355, 2.36 , 2.365, 2.37 , 2.375, 2.38 ,
        2.385, 2.39 , 2.395, 2.4  , 2.405, 2.41 , 2.415, 2.42 , 2.425,
        2.43 , 2.435, 2.44 , 2.445, 2.45 , 2.455, 2.46, 2.485, 2.51, 2.535, 2.56, 2.6, 2.8, 3. , 3.2, 3.4, 3.6, 3.8, 4.}};

    return dc::Vector2(velocity_array[i] * std::sin(std::atan(-(j - 93) * 0.0254 / 38.405)), velocity_array[i] * std::cos(std::atan(-(j - 93) * 0.0254 / 38.405)));
}


std::vector<torch::jit::IValue> GameStateToInput(std::vector<dc::GameState> game_states, dc::GameSetting game_setting)
{
    std::vector<torch::jit::IValue> inputs;

    torch::Tensor sheet = torch::zeros({static_cast<int>(game_states.size()), 2, 27*12+12, 12*15+7}).to(device);
    torch::Tensor end = torch::zeros({static_cast<int>(game_states.size()), 1}).to(device);
    torch::Tensor score = torch::zeros({static_cast<int>(game_states.size()), 1}).to(device);
    torch::Tensor shot = torch::zeros({static_cast<int>(game_states.size()), 1}).to(device);

    for (size_t k=0; k < game_states.size(); ++k){
        int i = static_cast<int>(k);
        if (game_states[i].IsGameOver()) continue;

        shot.index({i, 0}) = (game_states[i].kShotPerEnd - game_states[i].shot) / 16.f;
        score.index({i, 0}) = (static_cast<float>(game_states[i].GetTotalScore(game_states[i].hammer)) - static_cast<float>(game_states[i].GetTotalScore(dc::GetOpponentTeam(game_states[i].hammer)))) * 0.1f + 0.5f;
        if (game_states[i].end < game_setting.max_end){
            end.index({i, 0}) = (game_setting.max_end - game_states[i].end) / 10.f;
        } else {
            end.index({i, 0}) = 0.1f;
        }
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
    inputs.push_back(end);
    inputs.push_back(score);
    inputs.push_back(shot);

    return inputs;
}

// policyのショット効果のないショットを除くフィルターを作成
// ガードゾーン、ハウスに止まるショットと、シート状にあるストーンに干渉するショットのみを残す
torch::Tensor createFilter(dc::GameState game_state, dc::GameSetting game_setting)
{
    torch::Tensor filt = torch::zeros({2, 50, 12*15+7}).to("cpu");

    int min_velocity = 1;
    if (game_state.shot+1 == game_state.kShotPerEnd) min_velocity = 13; // ラストエンドはハウスに届かないショットを投げても意味がない

    for (auto i=0; i < 50; ++i){
        for (auto j=0; j < 187; ++j){
            if ((min_velocity <= i) && (i < 37) && (j < 94)) filt.index({1, i, j}) = 1;
            if ((min_velocity <= i) && (i < 37) && (j >= 93)) filt.index({0, i, j}) = 1;
        }
    }


    for (size_t team_stone_idx = 0; team_stone_idx < game_state.kShotPerEnd / 2; ++team_stone_idx) {
        auto const& stone_hammer = game_state.stones[static_cast<size_t>(game_state.hammer)][team_stone_idx];
        auto const& stone_nohammer = game_state.stones[static_cast<size_t>(dc::GetOpponentTeam(game_state.hammer))][team_stone_idx];
        if (stone_hammer) {
            std::pair <int, int> pixel = PositionToPixel(stone_hammer->position);
            // std::cout << pixel << std::endl;
            for (auto i=0; i < 50; ++i){
                for (auto j=0; j < 187; ++j){
                    if ((4*(i - 50) >= j - pixel.second + (pixel.first - 252)/20) && (4*(i - 50) <= j - pixel.second + (pixel.first - 252)/20 + 40)) {
                        filt.index({1, i, j}) = 1;
                    }
                    if ((-4*(i - 50) <= j - (187 - (pixel.second - (pixel.first - 252)/20))) && (-4*(i - 50) >= j - (187 - (pixel.second - (pixel.first - 252)/20 - 40)))) {
                        filt.index({0, i, j}) = 1;
                    }
                }
            }
        }
        if (stone_nohammer) {
            std::pair <int, int> pixel = PositionToPixel(stone_nohammer->position);
            // std::cout << pixel << std::endl;
            for (auto i=0; i < 50; ++i){
                for (auto j=0; j < 187; ++j){
                    if ((4*(i - 50) >= j - pixel.second + (pixel.first - 252)/20) && (4*(i - 50) <= j - pixel.second + (pixel.first - 252)/20 + 40)) {
                        filt.index({1, i, j}) = 1;
                    }
                    if ((-4*(i - 50) <= j - (187 - (pixel.second - (pixel.first - 252)/20))) && (-4*(i - 50) >= j - (187 - (pixel.second - (pixel.first - 252)/20 - 40)))) {
                        filt.index({0, i, j}) = 1;
                    }
                }
            }
        }
    }

    return filt;
}


/// \brief サーバーから送られてきた試合設定が引数として渡されるので，試合前の準備を行います．
///
/// 引数 \p player_order を編集することでプレイヤーのショット順を変更することができます．各プレイヤーの情報は \p player_factories に格納されています．
/// 補足：プレイヤーはショットのブレをつかさどります．プレイヤー数は4で，0番目は0, 1投目，1番目は2, 3投目，2番目は4, 5投目，3番目は6, 7投目を担当します．
///
/// この処理中の思考時間消費はありません．試合前に時間のかかる処理を行う場合この中で行うべきです．
///
/// \param team この思考エンジンのチームID．
///     Team::k0 の場合，最初のエンドの先攻です．
///     Team::k1 の場合，最初のエンドの後攻です．
///
/// \param game_setting 試合設定．
///     この参照はOnInitの呼出し後は無効になります．OnInitの呼出し後にも参照したい場合はコピーを作成してください．
///
/// \param simulator_factory 試合で使用されるシミュレータの情報．
///     未対応のシミュレータの場合 nullptr が格納されます．
///
/// \param player_factories 自チームのプレイヤー情報．
///     未対応のプレイヤーの場合 nullptr が格納されます．
///
/// \param player_order 出力用引数．
///     プレイヤーの順番(デフォルトで0, 1, 2, 3)を変更したい場合は変更してください．
void OnInit(
    dc::Team team,
    dc::GameSetting const& game_setting,
    std::unique_ptr<dc::ISimulatorFactory> simulator_factory,
    std::array<std::unique_ptr<dc::IPlayerFactory>, 4> player_factories,
    std::array<size_t, 4> & player_order)
{
    // TODO AIを作る際はここを編集してください
    g_team = team;

    torch::NoGradGuard no_grad; 

    device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        device = torch::kCUDA;
    }   
    // Deserialize the ScriptModule from a file using torch::jit::load().
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("model/traced_curling_cnn_gat2023.pt", device);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }


    for (unsigned i = 0; i < 1; ++i) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({1, 2, 27*12+12, 12*15+7}).to(device));
        inputs.push_back(torch::rand({1, 1}).to(device));
        inputs.push_back(torch::rand({1, 1}).to(device));
        inputs.push_back(torch::rand({1, 1}).to(device));

        // Execute the model and turn its output into a tensor.
        auto outputs = module.forward(inputs).toTuple();
        torch::Tensor out1 = outputs->elements()[0].toTensor().to(torch::kCPU);
        torch::Tensor out2 = outputs->elements()[1].toTensor().to(torch::kCPU); 
        torch::Tensor out3 = outputs->elements()[2].toTensor().to(torch::kCPU);

    }
    c10::cuda::CUDACachingAllocator::emptyCache();

    // 非対応の場合は シミュレータFCV1を使用する．
    g_team = team;
    g_game_setting = game_setting;
    if (simulator_factory) {
        g_simulator = simulator_factory->CreateSimulator();
        for (unsigned i = 0; i < nBatchSize; ++i) {
            g_simulators[i] = simulator_factory->CreateSimulator();
        }
    } else {
        g_simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
        for (unsigned i = 0; i < nBatchSize; ++i) {
            g_simulators[i] = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
        }
    }
    g_simulator_storage = g_simulator->CreateStorage();

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

    limit = g_game_setting.thinking_time[0] * 0.7 / 80.;

    int dummy = 0;

    // #pragma omp parallel for
    // for (auto i=0; i < 2000000; ++i) {
    //     ++dummy;
    // }
}



/// \brief 自チームのターンに呼ばれます．返り値として返した行動がサーバーに送信されます．
///
/// \param game_state 現在の試合状況．
///     この参照は関数の呼出し後に無効になりますので，関数呼出し後に参照したい場合はコピーを作成してください．
///
/// \return 選択する行動．この行動が自チームの行動としてサーバーに送信されます．
dc::Move OnMyTurn(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
    auto start = std::chrono::system_clock::now();

    dc::GameState current_game_state = game_state;

    // std::cout << (current_game_state.kShotPerEnd - current_game_state.shot) / 16.f << std::endl;
    // std::cout << (static_cast<float>(current_game_state.GetTotalScore(current_game_state.hammer)) - static_cast<float>(current_game_state.GetTotalScore(dc::GetOpponentTeam(current_game_state.hammer)))) * 0.1f + 0.5f << std::endl;
    // std::cout << (g_game_setting.max_end - current_game_state.end) / 10.f << std::endl;

    // auto inputs = GameStateToInput({current_game_state}, g_game_setting);

    // std::cout << inputs[1] << "  " << inputs[2] << "   " << inputs[3] << std::endl;
    // for (auto i=0; i < 12*28; ++i){
    //     for (auto j=0; j < 187; ++j){
    //         if (inputs[0].toTensor()[0][0][i][j].item<int>() + inputs[0].toTensor()[0][1][i][j].item<int>() > 0) std::cout << i << "  " << j << std::endl;
    //     }
    // }    

    torch::NoGradGuard no_grad; 

    // Create a vector of inputs.
    auto current_outputs = module.forward(GameStateToInput({current_game_state}, g_game_setting)).toTuple();

    auto policy = F::softmax(current_outputs->elements()[0].toTensor().reshape({1, 18700}).to(torch::kCPU), 1);

    torch::Tensor filt = createFilter(current_game_state, g_game_setting);


    std::array<int, nCandidate> indices_copy;
    std::array<dc::moves::Shot, nCandidate> shots;
    std::array<dc::Vector2, nCandidate> velocity;

    if (current_game_state.shot < 0){ // random shot 
        auto indices = std::get<1>(torch::topk(torch::rand({1, 18700}) * filt.reshape({1, 18700}), 1));

        int i = 0;
        indices_copy[i] = indices.index({0, i}).item<int>();
        velocity[i] = PixelToVelocity(indices_copy[i] % (50 * (12*15+7)) / (12*15+7), indices_copy[i] % (50 * (12*15+7)) % (12*15+7));

        if (indices_copy[i] / (50 * (12*15+7)) == 0) shots[i] = {velocity[i], dc::moves::Shot::Rotation::kCW};
        else if (indices_copy[i] / (50 * (12*15+7)) == 1) shots[i] = {velocity[i], dc::moves::Shot::Rotation::kCCW};
        else std::cerr << "shot error!";

        return shots[i];
    } else {

        // int idx = torch::argmax(policy[0]).item().to<int>();
        // int idx = torch::argmax(torch::rand({2, 50, 187})).item().to<int>();
        auto indices = std::get<1>(torch::topk((policy + torch::randn({1, 18700}) * 2e-4) * filt.reshape({1, 18700}), nCandidate));
        // auto indices = std::get<1>(torch::topk(torch::rand({1, 18700}) * filt.reshape({1, 18700}), nCandidate)); // random selection

        // std::cout << idx << std::endl;



        #pragma omp parallel for
        for (auto i = 0; i < nCandidate; ++i) {   
            indices_copy[i] = indices.index({0, i}).item<int>();
            velocity[i] = PixelToVelocity(indices_copy[i] % (50 * (12*15+7)) / (12*15+7), indices_copy[i] % (50 * (12*15+7)) % (12*15+7));

            if (indices_copy[i] / (50 * (12*15+7)) == 0) shots[i] = {velocity[i], dc::moves::Shot::Rotation::kCW};
            else if (indices_copy[i] / (50 * (12*15+7)) == 1) shots[i] = {velocity[i], dc::moves::Shot::Rotation::kCCW};
            else std::cerr << "shot error!";
        }


        auto & current_player = *g_players[game_state.shot / 4];

        g_simulator->Save(*g_simulator_storage);

        std::vector<dc::GameState> temp_game_states;
        temp_game_states.resize(nBatchSize);
        std::array<dc::Move, nBatchSize> temp_moves;

        torch::Tensor prob_ave = torch::zeros({nCandidate});

        int count = 0;
        auto now = std::chrono::system_clock::now();
        while ((count < nCandidate * nSimulation) && (now - start < limit)){
            #pragma omp parallel for
            for (auto i = 0; i < nBatchSize; ++i) {
                temp_game_states[i] = current_game_state;
                temp_moves[i] = shots[(count + i)/nSimulation];
                g_simulators[i]->Load(*g_simulator_storage);

                dc::ApplyMove(g_game_setting, *g_simulators[i],
                    current_player, temp_game_states[i], temp_moves[i], std::chrono::milliseconds(0));
            }

            // auto inputs = GameStateToInput(temp_game_states, g_game_setting);
            auto outputs = module.forward(GameStateToInput(temp_game_states, g_game_setting)).toTuple();

            torch::Tensor prob = torch::sigmoid(outputs->elements()[1].toTensor().to(torch::kCPU));

            // std::cout << inputs[1] << inputs[2] << inputs[3] << std::endl;

            if (g_team != current_game_state.hammer) prob = torch::ones({prob.sizes()[0], 1})-prob;

            if (current_game_state.shot+1 == current_game_state.kShotPerEnd){
                for (auto i=0; i < nBatchSize; ++i){
                    if (temp_game_states[i].IsGameOver()){
                        prob.index({i, 0}) = g_team == temp_game_states[i].game_result->winner;
                    } else if (temp_game_states[i].hammer == dc::GetOpponentTeam(g_team)) {
                        prob.index({i, 0}) = 1 - prob.index({i, 0});
                    }

                    // std::cout << dc::ToString(temp_game_states[i].hammer) << "  " << prob.index({i, 0}).item() << std::endl;
                }
            }

            for (auto i=0; i < nBatchSize / nSimulation; ++i){
                prob_ave[count/nSimulation+i] = torch::mean(prob.reshape({nBatchSize / nSimulation, nSimulation}), 1)[i];
            }

            count += nBatchSize;
            now = std::chrono::system_clock::now();
        }

        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        std::cout << count << " simulations in " << msec << "msec" << std::endl;

        // for (auto i=0; i < 8; ++i){
        //     std::cout << indices[0][i].item() << "   " << prob_ave[i].item<float>() << std::endl;
        // }

        return shots[torch::argmax(prob_ave).item().to<int>()];
    }

    // コンシードを行う場合
    // return dc::moves::Concede();
}



/// \brief 相手チームのターンに呼ばれます．AIを作る際にこの関数の中身を記述する必要は無いかもしれません．
///
/// ひとつ前の手番で自分が行った行動の結果を見ることができます．
///
/// \param game_state 現在の試合状況．
///     この参照は関数の呼出し後に無効になりますので，関数呼出し後に参照したい場合はコピーを作成してください．
void OnOpponentTurn(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
    // dc::GameState current_game_state = game_state;

    // Create a vector of inputs.
    // auto outputs = module.forward(GameStateToInput({current_game_state}, g_game_setting)).toTuple();
}



/// \brief ゲームが正常に終了した際にはこの関数が呼ばれます．
///
/// \param game_state 試合終了後の試合状況．
///     この参照は関数の呼出し後に無効になりますので，関数呼出し後に参照したい場合はコピーを作成してください．
void OnGameOver(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください

    if (game_state.game_result->winner == g_team) {
        std::cout << "won the game" << std::endl;
    } else {
        std::cout << "lost the game" << std::endl;
    }
}



} // unnamed namespace



int main(int argc, char const * argv[])
{
    using boost::asio::ip::tcp;
    using nlohmann::json;

    // TODO AIの名前を変更する場合はここを変更してください．
    constexpr auto kName = "CurlingCNN";

    constexpr int kSupportedProtocolVersionMajor = 1;

    try {
        if (argc != 3) {
            std::cerr << "Usage: command <host> <port>" << std::endl;
            return 1;
        }

        boost::asio::io_context io_context;

        tcp::socket socket(io_context);
        tcp::resolver resolver(io_context);
        boost::asio::connect(socket, resolver.resolve(argv[1], argv[2]));  // 引数のホスト，ポートに接続します．

        // ソケットから1行読む関数です．バッファが空の場合，新しい行が来るまでスレッドをブロックします．
        auto read_next_line = [&socket, input_buffer = std::string()] () mutable {
            // read_untilの結果，input_bufferに複数行入ることがあるため，1行ずつ取り出す処理を行っている
            if (input_buffer.empty()) {
                boost::asio::read_until(socket, boost::asio::dynamic_buffer(input_buffer), '\n');
            }
            auto new_line_pos = input_buffer.find_first_of('\n');
            auto line = input_buffer.substr(0, new_line_pos + 1);
            input_buffer.erase(0, new_line_pos + 1);
            return line;
        };

        // コマンドが予期したものかチェックする関数です．
        auto check_command = [] (nlohmann::json const& jin, std::string_view expected_cmd) {
            auto const actual_cmd = jin.at("cmd").get<std::string>();
            if (actual_cmd != expected_cmd) {
                std::ostringstream buf;
                buf << "Unexpected cmd (expected: \"" << expected_cmd << "\", actual: \"" << actual_cmd << "\")";
                throw std::runtime_error(buf.str());
            }
        };

        dc::Team team = dc::Team::kInvalid;

        // [in] dc
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "dc");

            auto const& jin_version = jin.at("version");
            if (jin_version.at("major").get<int>() != kSupportedProtocolVersionMajor) {
                throw std::runtime_error("Unexpected protocol version");
            }

            std::cout << "[in] dc" << std::endl;
            std::cout << "  game_id  : " << jin.at("game_id").get<std::string>() << std::endl;
            std::cout << "  date_time: " << jin.at("date_time").get<std::string>() << std::endl;
        }

        // [out] dc_ok
        {
            json const jout = {
                { "cmd", "dc_ok" },
                { "name", kName }
            };
            auto const output_message = jout.dump() + '\n';
            boost::asio::write(socket, boost::asio::buffer(output_message));

            std::cout << "[out] dc_ok" << std::endl;
            std::cout << "  name: " << kName << std::endl;
        }

        // dc::GameState game_state;

        // [in] is_ready
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "is_ready");

            if (jin.at("game").at("rule").get<std::string>() != "normal") {
                throw std::runtime_error("Unexpected rule");
            }

            team = jin.at("team").get<dc::Team>();

            auto const game_setting = jin.at("game").at("setting").get<dc::GameSetting>();

            auto const& jin_simulator = jin.at("game").at("simulator");
            std::unique_ptr<dc::ISimulatorFactory> simulator_factory;
            try {
                simulator_factory = jin_simulator.get<std::unique_ptr<dc::ISimulatorFactory>>();
            } catch (std::exception & e) {
                std::cout << "Exception: " << e.what() << std::endl;
            }

            auto const& jin_player_factories = jin.at("game").at("players").at(dc::ToString(team));
            std::array<std::unique_ptr<dc::IPlayerFactory>, 4> player_factories;
            for (size_t i = 0; i < 4; ++i) {
                std::unique_ptr<dc::IPlayerFactory> player_factory;
                try {
                    player_factory = jin_player_factories[i].get<std::unique_ptr<dc::IPlayerFactory>>();
                } catch (std::exception & e) {
                    std::cout << "Exception: " << e.what() << std::endl;
                }
                player_factories[i] = std::move(player_factory);
            }

            std::cout << "[in] is_ready" << std::endl;
        
        // [out] ready_ok

            std::array<size_t, 4> player_order{ 0, 1, 2, 3 };
            OnInit(team, game_setting, std::move(simulator_factory), std::move(player_factories), player_order);

            json const jout = {
                { "cmd", "ready_ok" },
                { "player_order", player_order }
            };
            auto const output_message = jout.dump() + '\n';
            boost::asio::write(socket, boost::asio::buffer(output_message));

            std::cout << "[out] ready_ok" << std::endl;
            std::cout << "  player order: " << jout.at("player_order").dump() << std::endl;
        }

        // [in] new_game
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "new_game");

            std::cout << "[in] new_game" << std::endl;
            std::cout << "  team 0: " << jin.at("name").at("team0") << std::endl;
            std::cout << "  team 1: " << jin.at("name").at("team1") << std::endl;
        }

        dc::GameState game_state;

        while (true) {
            // [in] update
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "update");

            game_state = jin.at("state").get<dc::GameState>();

            std::cout << "[in] update (end: " << int(game_state.end) << ", shot: " << int(game_state.shot) << ")" << std::endl;

            // if game was over
            if (game_state.game_result) {
                break;
            }

            if (game_state.GetNextTeam() == team) { // my turn
                // [out] move
                auto move = OnMyTurn(game_state);
                json jout = {
                    { "cmd", "move" },
                    { "move", move }
                };
                auto const output_message = jout.dump() + '\n';
                boost::asio::write(socket, boost::asio::buffer(output_message));
                
                c10::cuda::CUDACachingAllocator::emptyCache();

                std::cout << "[out] move" << std::endl;
                if (std::holds_alternative<dc::moves::Shot>(move)) {
                    dc::moves::Shot const& shot = std::get<dc::moves::Shot>(move);
                    std::cout << "  type    : shot" << std::endl;
                    std::cout << "  velocity: [" << shot.velocity.x << ", " << shot.velocity.y << "]" << std::endl;
                    std::cout << "  rotation: " << (shot.rotation == dc::moves::Shot::Rotation::kCCW ? "ccw" : "cw") << std::endl;
                } else if (std::holds_alternative<dc::moves::Concede>(move)) {
                    std::cout << "  type: concede" << std::endl;
                }

            } else { // opponent turn
                OnOpponentTurn(game_state);
            }
        }

        // [in] game_over
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "game_over");

            std::cout << "[in] game_over" << std::endl;
        }

        // 終了．
        OnGameOver(game_state);

    } catch (std::exception & e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 0;
}
