#ifndef SKIP_HPP
#define SKIP_HPP

#include <vector>

#include "uctnode.hpp"
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;

const int nSimulation = 4; // 1つのショットに対する誤差を考慮したシミュレーション回数
const int nBatchSize = 200; // CNNで推論するときのバッチサイズ
const int nLoop = 1000; // 
const int nCandidate = 10000; // シミュレーションするショットの最大数。制限時間でシミュレーションできる数よりも十分大きく取る

class Skip
{
    public:
        Skip(torch::jit::script::Module, dc::GameSetting,
            std::array<std::shared_ptr<dc::ISimulator>, nLoop>, std::array<std::shared_ptr<dc::IPlayer>, 4>, std::chrono::duration<double>, 
            std::vector<std::vector<double>>, torch::Device);

        float search(std::shared_ptr<UctNode>, int);

        void updateNode(std::shared_ptr<UctNode>, int, float);

        void SimulateMove(std::shared_ptr<UctNode>, int, int);
        torch::Tensor EvaluateGameState(std::vector<dc::GameState>, dc::GameSetting);
        void EvaluateQueue();

        dc::Move command(const dc::GameState&);

    private:
        torch::jit::script::Module module;
        dc::GameSetting g_game_setting;
        std::array<std::shared_ptr<dc::ISimulator>, nLoop> g_simulators;
        std::array<std::shared_ptr<dc::IPlayer>, 4> g_players;
        std::chrono::duration<double> limit;
        std::vector<std::vector<double>> win_table;
        torch::Device device;

        std::vector<std::shared_ptr<UctNode>> queue_evaluate;
        std::vector<int> queue_simulate;

        std::array<std::shared_ptr<UctNode>, nLoop> queue_create_child;
        std::array<int, nLoop> queue_create_child_index;
        std::array<bool, nLoop> flag_create_child;

        std::array<dc::GameState, nLoop> temp_game_states;



};

#endif // SKIP_HPP