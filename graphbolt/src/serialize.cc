/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/serialize.cc
 * @brief Source file of serialize.
 */

#include <graphbolt/serialize.h>
#include <torch/torch.h>

#include "./random.h"
namespace torch {

serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    graphbolt::sampling::FusedCSCSamplingGraph& graph) {
  graph.Load(archive);
  return archive;
}

serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const graphbolt::sampling::FusedCSCSamplingGraph& graph) {
  graph.Save(archive);
  return archive;
}

}  // namespace torch

namespace {
void dummy() {
  std::cout << graphbolt::RandomEngine::ThreadLocal()->manual_seed.value_or(0)
            << std::endl;
}
}  // namespace

namespace graphbolt {

torch::IValue read_from_archive(
    torch::serialize::InputArchive& archive, const std::string& key) {
  torch::IValue data;
  archive.read(key, data);
  return data;
}

}  // namespace graphbolt
