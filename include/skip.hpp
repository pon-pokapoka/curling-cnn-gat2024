#ifndef SKIP_HPP
#define SKIP_HPP

#include <vector>

#include "uctnode.hpp"
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;

const int nSimulation = 4; // 1つのショットに対する誤差を考慮したシミュレーション回数
const int nBatchSize = 1024; // CNNで推論するときのバッチサイズ
const int nLoop = 2048; // 
// const int nCandidate = 10000; // シミュレーションするショットの最大数。制限時間でシミュレーションできる数よりも十分大きく取る

class Skip
{
    public:
        Skip();

        void OnInit(dc::Team const, dc::GameSetting const&,     std::unique_ptr<dc::ISimulatorFactory>,     std::array<std::unique_ptr<dc::IPlayerFactory>, 4>, std::array<size_t, 4> & );

        float search(UctNode*, int);
        void searchById(UctNode*, int, int);

        void updateNodes();
        void updateParent(UctNode*, float);

        void SimulateMove(UctNode*, int, int);
        std::vector<float> EvaluateGameState(std::vector<dc::GameState>, dc::GameSetting);
        void EvaluateQueue();

        dc::Move command(const dc::GameState&);

    private:
        dc::Team team;
        torch::jit::script::Module module;
        dc::GameSetting g_game_setting;
        std::array<std::unique_ptr<dc::ISimulator>, nLoop> g_simulators;
        std::unique_ptr<dc::ISimulatorStorage> g_simulator_storage;
        std::array<std::unique_ptr<dc::IPlayer>, 4> g_players;
        std::chrono::duration<double> limit;
        std::vector<std::vector<double>> win_table;
        torch::Device device;

        std::vector<UctNode*> queue_evaluate;
        std::vector<int> queue_simulate;

        std::array<UctNode*, nLoop> queue_create_child;
        std::array<int, nLoop> queue_create_child_index;
        std::array<bool, nLoop> flag_create_child;

        std::array<dc::GameState, nLoop> temp_game_states;

        int kShotPerEnd;

// torch::jit::script::Module module; // モデル

// // シミュレーション用変数
// dc::GameSetting g_game_setting;
// std::unique_ptr<dc::ISimulator> g_simulator;
// std::array<std::unique_ptr<dc::ISimulator>, nLoop> g_simulators;
// std::array<std::unique_ptr<dc::IPlayer>, 4> g_players;

// std::chrono::duration<double> limit; // 考慮時間制限




};

#endif // SKIP_HPP