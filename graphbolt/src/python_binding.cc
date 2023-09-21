/**
 *  Copyright (c) 2023 by Contributors
 * @file python_binding.cc
 * @brief Graph bolt library Python binding.
 */

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/serialize.h>
#include <graphbolt/unique_and_compact.h>

namespace graphbolt {
namespace sampling {

TORCH_LIBRARY(graphbolt, m) {
  m.class_<SampledSubgraph>("SampledSubgraph")
      .def(torch::init<>())
      .def_readwrite("indptr", &SampledSubgraph::indptr)
      .def_readwrite("indices", &SampledSubgraph::indices)
      .def_readwrite(
          "reverse_row_node_ids", &SampledSubgraph::reverse_row_node_ids)
      .def_readwrite(
          "reverse_column_node_ids", &SampledSubgraph::reverse_column_node_ids)
      .def_readwrite("reverse_edge_ids", &SampledSubgraph::reverse_edge_ids)
      .def_readwrite("type_per_edge", &SampledSubgraph::type_per_edge)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<SampledSubgraph>& self)
              -> std::vector<torch::Tensor> { return self->GetState(); },
          // __setstate__
          [](std::vector<torch::Tensor> state)
              -> c10::intrusive_ptr<SampledSubgraph> {
            auto g = c10::make_intrusive<SampledSubgraph>();
            g->SetState(state);
            return g;
          });
  m.class_<CSCSamplingGraph>("CSCSamplingGraph")
      .def("num_nodes", &CSCSamplingGraph::NumNodes)
      .def("num_edges", &CSCSamplingGraph::NumEdges)
      .def("csc_indptr", &CSCSamplingGraph::CSCIndptr)
      .def("indices", &CSCSamplingGraph::Indices)
      .def("node_type_offset", &CSCSamplingGraph::NodeTypeOffset)
      .def("type_per_edge", &CSCSamplingGraph::TypePerEdge)
      .def("edge_attributes", &CSCSamplingGraph::EdgeAttributes)
      .def("in_subgraph", &CSCSamplingGraph::InSubgraph)
      .def("sample_neighbors", &CSCSamplingGraph::SampleNeighbors)
      .def(
          "sample_negative_edges_uniform",
          &CSCSamplingGraph::SampleNegativeEdgesUniform)
      .def("copy_to_shared_memory", &CSCSamplingGraph::CopyToSharedMemory)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<CSCSamplingGraph>& self)
              -> torch::Dict<
                  std::string, torch::Dict<std::string, torch::Tensor>> {
            return self->GetState();
          },
          // __setstate__
          [](torch::Dict<std::string, torch::Dict<std::string, torch::Tensor>>
                 state) -> c10::intrusive_ptr<CSCSamplingGraph> {
            auto g = c10::make_intrusive<CSCSamplingGraph>();
            g->SetState(state);
            return g;
          });
  m.def("from_csc", &CSCSamplingGraph::FromCSC);
  m.def("load_csc_sampling_graph", &LoadCSCSamplingGraph);
  m.def("save_csc_sampling_graph", &SaveCSCSamplingGraph);
  m.def("load_from_shared_memory", &CSCSamplingGraph::LoadFromSharedMemory);
  m.def("unique_and_compact", &UniqueAndCompact);
}

}  // namespace sampling
}  // namespace graphbolt
