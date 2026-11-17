"""Delayed-completion/abort ownership traces for the same demotion guards used by Model."""

import os
import pathlib
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def demotion_test(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("a C++20 compiler is needed")
    out = tmp_path_factory.mktemp("async-kv")
    source = out / "demotion.cc"
    source.write_text(r'''
#include <cassert>
#include <cstring>
#include <random>
#include <string>
#include "kv_demotion.h"
using namespace q27;
int main(int argc,char** argv) {
  const std::string mode=argv[1];
  HostKvPages host;
  host.init(4,64,[](size_t n){return new char[n];},[](char* p){delete[] p;});
  PagePool gpu; gpu.init(5);
  KvDemotion pending;
  if (mode=="evicted_source") {
    int p=gpu.alloc(), h=host.alloc();
    std::vector<Checkpoint> cs(1); cs[0].pages={p};
    pending.hold(p,h,gpu);
    assert(demotion_batch(cs,gpu).empty()); // cannot queue the same source twice
    gpu.unref(p); cs.clear(); // checkpoint expires before DMA completion
    assert(gpu.refs(p)==1 && host.refs(h)==1 && gpu.free_pages()==3);
    for (int i=0;i<3;++i) assert(gpu.alloc()!=p);
    assert(gpu.alloc()==-1); // neither the old source nor destination can be reused
    std::memset(host.data(h),17,64); // emulate copy fence
    pending.publish_after_fence(cs,[&](int p){gpu.unref(p);},host);
    pending.release_after_fence(gpu,host);
    assert(gpu.alloc()==p && host.used()==0 && pending.empty());
  } else if (mode=="shared_new_active") {
    int p=gpu.alloc(), h=host.alloc(); gpu.ref(p);
    std::vector<Checkpoint> cs(2);
    for (auto& c:cs) { c.pages={p}; c.slot=7; c.ring=true; c.ids={1,2,3}; }
    pending.hold(p,h,gpu);
    gpu.ref(p); // a restored sequence acquires the immutable full source while copying
    std::memset(host.data(h),31,64);
    pending.publish_after_fence(cs,[&](int p){gpu.unref(p);},host);
    pending.release_after_fence(gpu,host);
    assert(gpu.refs(p)==1 && host.refs(h)==2);
    for (const auto& c:cs) assert(c.pages[0]==h && c.slot==7 && c.ring && c.ids.size()==3);
    assert(host.data(h)[63]==31);
    gpu.unref(p);
  } else if (mode=="abort") {
    std::vector<Checkpoint> cs(1);
    for (int i=0;i<3;++i) {
      int p=gpu.alloc(),h=host.alloc(); cs[0].pages.push_back(p); pending.hold(p,h,gpu);
    }
    // Queueing failed part way through. Drain the stream, then release WITHOUT publishing.
    pending.release_after_fence(gpu,host);
    assert(host.used()==0 && pending.empty() && gpu.free_pages()==1);
    for (int p:cs[0].pages) assert(p>0 && gpu.refs(p)==1);
    assert(demotion_batch(cs,gpu).size()==3);
  } else if (mode=="random") {
    std::mt19937 rng(91);
    for (int trial=0;trial<10000;++trial) {
      assert(host.used()==0 && gpu.free_pages()==4 && pending.empty());
      int p=gpu.alloc(),h=host.alloc();
      std::vector<Checkpoint> cs(1+rng()%3);
      for (size_t i=0;i<cs.size();++i) { cs[i].pages={p}; if(i)gpu.ref(p); }
      pending.hold(p,h,gpu);
      bool active=rng()%2; if(active)gpu.ref(p);
      while (!cs.empty() && rng()%2) { gpu.unref(p); cs.pop_back(); }
      assert(gpu.refs(p)==int(cs.size())+1+active);
      std::memset(host.data(h),trial%127,64); // complete every copy before publish/abort
      bool publish=rng()%2;
      if(publish) pending.publish_after_fence(cs,[&](int p){gpu.unref(p);},host);
      pending.release_after_fence(gpu,host);
      assert(gpu.refs(p)==(publish?0:int(cs.size()))+active);
      assert(host.refs(h)==(publish?int(cs.size()):0));
      for (auto& c:cs) {
        if(c.pages[0]<0) { assert(host.data(c.pages[0])[0]==trial%127); host.unref(c.pages[0]); }
        else gpu.unref(c.pages[0]);
      }
      if(active)gpu.unref(p);
    }
  } else if (mode=="allocation_failure") {
    HostKvPages failed;
    failed.init(32,64,[](size_t)->char*{return nullptr;},[](char* p){delete[] p;});
    assert(failed.available()==32);
    assert(failed.alloc()==0 && failed.available()==0 && failed.failures()==1);
    assert(pending.empty());
  } else return 1;
}
''', encoding="utf-8")
    binary = out / "demotion"
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if os.environ.get("Q27_TEST_SANITIZERS") else []
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", *flags, "-I",
                    str(pathlib.Path(__file__).resolve().parents[1] / "src"), str(source), "-o", str(binary)],
                   check=True, capture_output=True)
    return binary


@pytest.mark.parametrize("case", ["evicted_source", "shared_new_active", "abort", "random", "allocation_failure"])
def test_inflight_demotion_ownership(demotion_test, case):
    result = subprocess.run([str(demotion_test), case], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
