#include "uctnode.hpp"

UctNode::UctNode():
parent_(nullptr),
game_state_(),
child_nodes_(),
child_move_indices_(),
evaluated_(false),
simulated_(false),
value_(-1)
{}

void UctNode::CreateChild(int index) {
    std::unique_ptr<UctNode> child(new UctNode());
    child->parent_ = this;
    child_nodes_.push_back(std::move(child));

    child_move_indices_.push_back(index);
}

// void UctNode::expandChild(int childData) {
//     std::unique_ptr<UctNode> newChild = new UctNode(childData);
//     addChild(newChild);
// }

// void UctNode::resetAsRoot() {
    // if (parent_ != nullptr) {
    //     parent_->removeChild(this);
    //     parent_ = nullptr;
    // }
// }

// void UctNode::removeChild(std::unique_ptr<UctNode> child) {
//     auto it = std::find(child_nodes_.begin(), child_nodes_.end(), child);
//     if (it != child_nodes_.end()) {
//         child_nodes_.erase(it);
//     }
// }

UctNode* UctNode::GetChild(int index)
{
    auto it = std::find(child_move_indices_.begin(), child_move_indices_.end(), index);

    return child_nodes_[it - child_move_indices_.begin()].get();
}

void UctNode::SetGameState(dc::GameState game_state)
{
    game_state_ = game_state;
    simulated_ = true;
}

dc::GameState UctNode::GetGameState()
{
    return game_state_;
}

void UctNode::SetPolicy(torch::Tensor policy)
{
    policy_ = policy;
}

void UctNode::SetFilter(torch::Tensor filter)
{
    filter_ = filter;
}

void UctNode::SetValue(float value)
{
    value_ = value;
}

void UctNode::SetEvaluatedResults(torch::Tensor policy, float value)
{
    policy_ = policy;
    value_ = value;
    evaluated_ = true;
}

void UctNode::SetSimulated()
{
    simulated_ = true;
}

void UctNode::SetEvaluated()
{
    evaluated_ = true;
}

bool UctNode::GetEvaluated()
{
    return evaluated_;
}

float UctNode::GetValue()
{
    return value_;
}

std::vector<std::unique_ptr<UctNode>> UctNode::GetChildNodes()
{
    return std::move(child_nodes_);
}

std::vector<int> UctNode::GetChildIndices()
{
    return child_move_indices_;
}
