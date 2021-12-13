/*!
 *  Copyright (c) 2019 by Contributors
 * \file tp_communicator.cc
 * \brief Tensorpipe Communicator for DGL distributed training.
 */

#include "tp_communicator.h"

#include <time.h>
#include <unistd.h>

#include <future>
#include <memory>
#include <utility>

#include "../rpc.h"

namespace dgl {
namespace rpc {

using namespace tensorpipe;

bool TPSender::ConnectReceiver(const std::string &addr, int recv_id) {
  if (pipes_.find(recv_id) != pipes_.end()) {
    LOG(WARNING) << "Duplicate recv_id[" << recv_id << "]. Ignoring...";
    return true;
  }
  std::shared_ptr<Pipe> pipe;
  pipe = context->connect(addr);
  auto done = std::make_shared<std::promise<bool>>();
  tensorpipe::Message tpmsg;
  tpmsg.metadata = "dglconnect";
  pipe->write(tpmsg, [done](const tensorpipe::Error &error) {
    if (error) {
      LOG(WARNING) << "Error occurred when write to pipe: " << error.what();
      done->set_value(false);
    } else {
      done->set_value(true);
    }
  });
  if (!done->get_future().get()) {
    LOG(WARNING) << "Failed to connect to receiver[" << addr << "].";
    return false;
  }
  pipes_[recv_id] = pipe;
  return true;
}

void TPSender::Send(const RPCMessage& msg, int recv_id) {
  auto pipe = pipes_[recv_id];
  tensorpipe::Message tp_msg;
  std::string* zerocopy_blob_ptr = &tp_msg.metadata;
  StreamWithBuffer zc_write_strm(zerocopy_blob_ptr, true);
  zc_write_strm.Write(msg);
  int32_t nonempty_ndarray_count = zc_write_strm.buffer_list().size();
  zerocopy_blob_ptr->append(reinterpret_cast<char*>(&nonempty_ndarray_count),
                            sizeof(int32_t));
  tp_msg.tensors.resize(nonempty_ndarray_count);
  // Hold the NDArray that ensure it's valid until write operation completes
  auto ndarray_holder = std::make_shared<std::vector<NDArray>>();
  ndarray_holder->resize(nonempty_ndarray_count);
  auto& buffer_list = zc_write_strm.buffer_list();
  for (size_t i = 0; i < buffer_list.size(); i++) {
    auto& ptr = buffer_list[i];
    (*ndarray_holder.get())[i] = ptr.tensor;
    tensorpipe::CpuBuffer cpu_buffer;
    cpu_buffer.ptr = ptr.data;
    tp_msg.tensors[i].buffer = cpu_buffer;
    tp_msg.tensors[i].length = ptr.size;
    if (ptr.size == 0) {
      LOG(FATAL) << "Cannot send a empty NDArray.";
    }
  }
  pipe->write(tp_msg,
              [ndarray_holder, recv_id](const tensorpipe::Error& error) {
                if (error) {
                  LOG(FATAL) << "Failed to send message to " << recv_id
                             << ". Details: " << error.what();
                }
              });
}

void TPSender::Finalize() {
  for (auto &&p : pipes_) {
    p.second->close();
  }
  pipes_.clear();
}

void TPReceiver::Finalize() {
  stop_wait_ = true;
  if (wait_thread_.joinable()) {
    wait_thread_.join();
  }
  for (auto &&p : pipes_) {
    p.second->close();
  }
  pipes_.clear();
}

bool TPReceiver::Wait(const std::string &addr, int num_sender, bool blocking) {
  if (wait_thread_.joinable()) {
    LOG(WARNING) << "TPReceiver::Wait() has been called already. Ignoring...";
    return false;
  }
  LOG(INFO) << "~~~~~~~~TPReceiver::Wait()";
  wait_thread_ = std::thread([this, addr]() {
    LOG(INFO) << "TPReceiver starts to wait on [" << addr << "].";
    listener = context->listen({addr});
    listen_booted_ = true;
    while (!stop_wait_) {
      auto pipe_prom = std::make_shared<std::promise<std::shared_ptr<Pipe>>>();
      listener->accept(
          [pipe_prom](const Error &error, std::shared_ptr<Pipe> pipe) {
            if (error) {
              LOG(WARNING) << error.what();
              pipe_prom->set_value(nullptr);
            } else {
              pipe_prom->set_value(std::move(pipe));
            }
          });
      auto &&fut = pipe_prom->get_future();
      std::future_status status;
      do {
        status = fut.wait_for(std::chrono::milliseconds(100));
      } while (status != std::future_status::ready && !stop_wait_);
      if (status != std::future_status::ready) {
        // stop waiting
        break;
      }
      std::shared_ptr<Pipe> pipe = fut.get();
      CHECK(pipe) << "Invalid pipe.";
      std::promise<bool> checkConnect;
      pipe->readDescriptor(
          [pipe, &checkConnect](const Error &error, Descriptor descriptor) {
            Allocation allocation;
            checkConnect.set_value(descriptor.metadata == "dglconnect");
            pipe->read(allocation, [](const Error &error) {});
          });
      auto &&check_fut = checkConnect.get_future();
      do {
        status = check_fut.wait_for(std::chrono::milliseconds(100));
      } while (status != std::future_status::ready && !stop_wait_);
      if (status != std::future_status::ready) {
        // stop waiting
        break;
      }
      CHECK(check_fut.get()) << "Invalid connect message.";
      pipes_[num_connected_] = pipe;
      ReceiveFromPipe(pipe, queue_);
      ++num_connected_;
    }
    listener->close();
    LOG(INFO) << "TPReceiver stops waiting on [" << addr << "].";
  });
  while (!listen_booted_) {
  }
  while (blocking && (num_sender != num_connected_)) {
  }
  return true;
}

void TPReceiver::ReceiveFromPipe(std::shared_ptr<Pipe> pipe,
                                 std::shared_ptr<RPCMessageQueue> queue) {
  pipe->readDescriptor([pipe, queue = std::move(queue)](const Error& error,
                                                        Descriptor descriptor) {
    if (error) {
      // Error may happen when the pipe is closed
      return;
    }
    Allocation allocation;
    CHECK_EQ(descriptor.payloads.size(), 0) << "Invalid DGL RPC Message";

    int tensorsize = descriptor.tensors.size();
    if (tensorsize > 0) {
      allocation.tensors.resize(tensorsize);
      for (size_t i = 0; i < descriptor.tensors.size(); i++) {
        tensorpipe::CpuBuffer cpu_buffer;
        cpu_buffer.ptr = new char[descriptor.tensors[i].length];
        allocation.tensors[i].buffer = cpu_buffer;
      }
    }
    pipe->read(
      allocation, [allocation, descriptor = std::move(descriptor),
                   queue = std::move(queue), pipe](const Error& error) {
        if (error) {
          // Because we always have a read event posted to the epoll,
          // Therefore when pipe is closed, error will be raised.
          // But this error is expected.
          // Other error is not expected. But we cannot identify the error with each
          // Other for now. Thus here we skip handling for all errors
          return;
        }

        char* meta_msg_begin = const_cast<char*>(&descriptor.metadata[0]);
        std::vector<void*> buffer_list(descriptor.tensors.size());
        for (size_t i = 0; i < descriptor.tensors.size(); i++) {
          buffer_list[i] = allocation.tensors[i].buffer.unwrap<CpuBuffer>().ptr;
        }
        StreamWithBuffer zc_read_strm(
          meta_msg_begin, descriptor.metadata.size() - sizeof(int32_t),
          buffer_list);
        RPCMessage msg;
        zc_read_strm.Read(&msg);
        queue->push(msg);
        TPReceiver::ReceiveFromPipe(pipe, queue);
      });
  });
}

void TPReceiver::Recv(RPCMessage* msg) { *msg = std::move(queue_->pop()); }

}  // namespace rpc
}  // namespace dgl
