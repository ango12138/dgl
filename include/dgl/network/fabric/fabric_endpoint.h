#pragma once

#include <arpa/inet.h>
// #include <control_endpoint.h>
#include <dgl/network/fabric/fabric_context.h>
#include <dgl/network/fabric/fabric_provider.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <dmlc/logging.h>
#include <netinet/in.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_tagged.h>

#include <chrono>
#include <memory>
#include <unordered_map>

namespace dgl {
namespace network {

void HandleCQError(struct fid_cq *cq);

const int kMaxConcurrentWorkRequest = 4224;  // 128 + 2048 * 2

struct FullFabricAddr {
  FabricAddr faddr;
  std::string readable_addr;
  fi_addr_t fiaddr;
};

class FabricEndpoint {
 public:
  FabricEndpoint() {}
  void Init(std::string prov_name) {
    if (!fabric_provider) {
      fabric_provider = std::make_shared<FabricProvider>("shm");
      fabric_ctx =
        std::unique_ptr<FabricContext>(new FabricContext(fabric_provider));
      CHECK(fabric_ctx != nullptr);
    }
  }

  FabricEndpoint(std::shared_ptr<FabricProvider> fabric_provider)
      : fabric_provider(fabric_provider) {
    fabric_ctx =
      std::unique_ptr<FabricContext>(new FabricContext(fabric_provider));
    CHECK(fabric_ctx != nullptr);
  }

  fi_addr_t AddPeerAddr(FabricAddr *addr) {
    fi_addr_t peer_addr;
    int ret =
      fi_av_insert(fabric_ctx->av.get(), addr->name, 1, &peer_addr, 0, nullptr);

    // CHECK_EQ(ret, 1) << "Call to fi_av_insert() failed. Return Code: " << ret
    //                  << ". ERROR: " << fi_strerror(-ret);

    // fi_av_straddr: human readable name
    FabricAddr readable_addr;
    fi_av_straddr(fabric_ctx->av.get(), addr->name, readable_addr.name,
                  &readable_addr.len);
    std::string readable_peer_addr =
      std::string(readable_addr.name, readable_addr.len);
    // LOG(INFO) << "Readable peer addr: " << readable_peer_addr;
    FullFabricAddr full_fi_addr = {
      .faddr = *addr, .readable_addr = readable_peer_addr, .fiaddr = peer_addr};
    client_ep.push_back(std::move(full_fi_addr));
    return peer_addr;
  };

  void Send(const void *buffer, size_t size, uint64_t tag, fi_addr_t peer_addr,
            bool sync = false) {
    while (true) {
      int ret = fi_send(fabric_ctx->ep.get(), buffer, size, nullptr, peer_addr,
                        nullptr);
      if (ret == -FI_EAGAIN) {
        // LOG(WARNING) << "fi_tsend: FI_EAGAIN";
      } else if (ret < 0) {
        check_err(ret, "Unable to do fi_send message");
      } else {
        break;
      }
    }
    if (sync) WaitCQ(1, fabric_ctx->txcq.get());

    // while (true) {
    //   int ret = fi_cq_read(fabric_ctx->cq.get(), &comp, 1);
    //   if (ret == 1) break;
    // }
  }

  void Recv(void *buffer, size_t size, uint64_t tag, fi_addr_t peer_addr,
            bool sync = false) {
    while (true) {
      int ret = fi_trecv(fabric_ctx->ep.get(), buffer, size, nullptr, peer_addr,
                         tag, 0, nullptr);
      if (ret == -FI_EAGAIN) {
        // no resources
        LOG(WARNING) << "fi_recv: FI_EAGAIN";
        continue;
      } else if (ret < 0) {
        check_err(ret, "Unable to do fi_recv message");
      }
      break;
    }
    if (sync) WaitCQ(1, fabric_ctx->rxcq.get());
    // while (true) {
    //   int ret = fi_cq_read(fabric_ctx->cq.get(), &comp, 1);
    //   // check_err(ret, "cq read");
    //   // if (comp.len > 0) break;
    //   if (ret == 1) break;
    // }
  }

  void WaitCQ(int64_t count, struct fid_cq *cq) {
    CHECK_EQ(count, 1) << "Only 1 is tested";
    struct fi_cq_tagged_entry cq_entries;
    while (true) {
      int ret = fi_cq_read(cq, &cq_entries, count);
      if (ret == -FI_EAGAIN) {
        return;
      } else if (ret == -FI_EAVAIL) {
        HandleCQError(cq);
      } else if (ret < 0) {
        check_err(ret, "fi_cq_read failed");
      } else {
        break;
      }
    }
  }

  // the name of the peer endpoint
  std::vector<FullFabricAddr> client_ep;

  struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];

  // Fabric Context
  std::unique_ptr<FabricContext> fabric_ctx;

  // Fabric info
  std::shared_ptr<FabricProvider> fabric_provider;
};  // namespace network

}  // namespace network
}  // namespace dgl
