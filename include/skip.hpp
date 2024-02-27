#ifndef SKIP_HPP
#define SKIP_HPP

#include <vector>

#include "uctnode.hpp"
#include "utility.hpp"
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;

class Skip
{
    public:
        Skip(torch::jit::script::Module, dc::GameSetting,
            std::vector<std::unique_ptr<dc::ISimulator>>);

        float search(std::unique_ptr<UctNode>, std::vector<std::unique_ptr<UctNode>>);

        void updateNode(std::unique_ptr<UctNode>, int, float);

        void SimulateMove(std::unique_ptr<UctNode>, int, int);
        void EvaluateGameState(std::vector<dc::GameState>, dc::GameSetting);
        void EvaluateQueue(std::vector<std::unique_ptr<UctNode>>);

        dc::Move command(const dc::GameState&);

    private:
        torch::jit::script::Module module;
        dc::GameSetting g_game_setting;
        std::vector<std::unique_ptr<dc::ISimulator>> g_simulators;
};

#endif // SKIP_HPP