#ifndef SKIP_HPP
#define SKIP_HPP

#include <vector>

#include "uctnode.hpp"
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;

const int nSimulation = 4; // 1つのショットに対する誤差を考慮したシミュレーション回数
const int nBatchSize = 200; // CNNで推論するときのバッチサイズ
const int nCandidate = 10000; // シミュレーションするショットの最大数。制限時間でシミュレーションできる数よりも十分大きく取る

class Skip
{
    public:
        Skip(torch::jit::script::Module, dc::GameSetting,
            std::array<std::unique_ptr<dc::ISimulator>, nBatchSize>, std::array<std::unique_ptr<dc::IPlayer>, 4>, std::chrono::duration<double>, 
            std::vector<std::vector<double>>, torch::Device);

        float search(std::unique_ptr<UctNode>);

        void updateNode(std::unique_ptr<UctNode>, int, float);

        void SimulateMove(std::unique_ptr<UctNode>, int, int);
        torch::Tensor EvaluateGameState(std::vector<dc::GameState>, dc::GameSetting);
        void EvaluateQueue();

        dc::Move command(const dc::GameState&);

    private:
        torch::jit::script::Module module;
        dc::GameSetting g_game_setting;
        std::array<std::unique_ptr<dc::ISimulator>, nBatchSize> g_simulators;
        std::array<std::unique_ptr<dc::IPlayer>, 4> g_players;
        std::chrono::duration<double> limit;
        std::vector<std::vector<double>> win_table;
        torch::Device device;

        std::vector<std::unique_ptr<UctNode>> queue_evaluate;


};

#endif // SKIP_HPP