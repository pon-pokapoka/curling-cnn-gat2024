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

#include "skip.hpp"
#include "readcsv.hpp"


namespace dc = digitalcurling3;
namespace F = torch::nn::functional;


namespace {


dc::Team g_team;  // 自身のチームID

torch::jit::script::Module module; // モデル

// シミュレーション用変数
dc::GameSetting g_game_setting;
std::unique_ptr<dc::ISimulator> g_simulator;
std::array<std::shared_ptr<dc::ISimulator>, nLoop> g_simulators;
std::unique_ptr<dc::ISimulatorStorage> g_simulator_storage;
std::array<std::shared_ptr<dc::IPlayer>, 4> g_players;

std::chrono::duration<double> limit; // 考慮時間制限

torch::Device device(torch::kCPU);

std::vector<std::vector<double>> win_table;

// policyのショットのうち効果のないショットを除くフィルターを作成
// ガードゾーン、ハウスに止まるショットと、シート上にあるストーンに干渉するショットのみを残す
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
        std::cout << "model loaded" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    // ここでCNNによる推論を行うことで、次回以降の速度が早くなる
    // 使うバッチサイズすべてで行っておく
    std::cout << "initial inference\n";
    for (auto i = 0; i < 10; ++i) {
        std::cout << ".";
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({nBatchSize, 18, 64, 16}).to(device));
        // inputs.push_back(torch::rand({nBatchSize, 1}).to(device));
        // inputs.push_back(torch::rand({nBatchSize, 1}).to(device));
        // inputs.push_back(torch::rand({nBatchSize, 1}).to(device));

        // Execute the model and turn its output into a tensor.
        auto outputs = module.forward(inputs).toTensor();
        torch::Tensor out1 = outputs.to(torch::kCPU);
        // torch::Tensor out2 = outputs->elements()[1].toTensor().to(torch::kCPU); 
        // torch::Tensor out3 = outputs->elements()[2].toTensor().to(torch::kCPU);
    }
    std::cout << "\n";
    for (auto i = 0; i < 10; ++i) {
        std::cout << ".";
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::rand({1, 18, 64, 16}).to(device));
        // inputs.push_back(torch::rand({1, 1}).to(device));
        // inputs.push_back(torch::rand({1, 1}).to(device));
        // inputs.push_back(torch::rand({1, 1}).to(device));

        // Execute the model and turn its output into a tensor.
        auto outputs = module.forward(inputs).toTensor();
        torch::Tensor out1 = outputs.to(torch::kCPU);
        // torch::Tensor out2 = outputs->elements()[1].toTensor().to(torch::kCPU); 
        // torch::Tensor out3 = outputs->elements()[2].toTensor().to(torch::kCPU);
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::cout << "\n";

    // シミュレータFCV1Lightを使用する．
    g_team = team;
    g_game_setting = game_setting;
    g_simulator = dc::simulators::SimulatorFCV1LightFactory().CreateSimulator();
    for (unsigned i = 0; i < nLoop; ++i) {
        g_simulators[i] = dc::simulators::SimulatorFCV1LightFactory().CreateSimulator();
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

    // 考慮時間制限
    // ショット数で等分するが、超過分を考慮して0.8倍しておく
    limit = g_game_setting.thinking_time[0] * 0.8 / 8. / g_game_setting.max_end;

    // ショットシミュレーションの動作確認
    // しなくて良い
    std::cout << "initial simulation\n";
    for (auto j = 0; j < 10; ++j) {
        std::cout << ".";
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
    std::cout << "\n";
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
    Skip skip(module, g_game_setting, g_simulators, g_players, limit, win_table, device);

    return skip.command(game_state);
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
                
                // GPUのキャッシュをクリア
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
