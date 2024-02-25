#include "uctnode.hpp"

UctNode::UctNode():
parent_(nullptr),
game_state_(),
child_nodes_(),
child_move_indices_(),
evaluated_(false),
simulated_(false),
{}

void UctNode::CreateChild(int index) {
    std::unique_ptr<UctNode> child(new UctNode());
    child->parent_ = this;
    child_nodes_.push_back(std::move(child));

    child_move_indices_.push_back(index);
}

void UctNode::expandChild(int childData) {
    UctNode* newChild = new UctNode(childData);
    addChild(newChild);
}

void UctNode::resetAsRoot() {
    if (parent != nullptr) {
        parent->removeChild(this);
        parent = nullptr;
    }
}

void UctNode::removeChild(UctNode* child) {
    auto it = std::find(childNodes.begin(), childNodes.end(), child);
    if (it != childNodes.end()) {
        childNodes.erase(it);
    }
}

UctNode* UctNode::GetChild(int index)
{
    return child_nodes_[index];
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

void UctNode::SetPolicy(Torch::Tensor policy)
{
    policy_ = policy;
}

void UctNode::SetFilter(Torch::Tensor filter)
{
    filter_ = filter;
}

void UctNode::SetValue(float value)
{
    value_ = value;
}

void UctNode::SetEvaluatedResults(Torch::Tensor policy, float value)
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

std::vector<std::unique_ptr<UctNode>> UctNode::GetChildNodes()
{
    return child_nodes_;
}

std::vector<int> UctNode::GetChildIndices()
{
    return child_move_indices_;
}
