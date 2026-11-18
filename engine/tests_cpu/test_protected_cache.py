"""Cache-policy traces on the actual C++ helpers, including capacity, ring and ownership boundaries."""

import os
import pathlib
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def policy_test(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("a C++20 compiler is needed")
    out = tmp_path_factory.mktemp("protected-cache")
    source = out / "policy.cc"
    source.write_text(r'''
#include <cassert>
#include <random>
#include <string>
#include "prefix_cache.h"
using namespace q27;
Checkpoint ck(int id, uint64_t time, int pages=4) {
  Checkpoint c; c.ids={id,id}; c.last_use=time; c.boundary=true; c.ring=true;
  for (int i=0;i<pages;++i) c.pages.push_back(-1-id*10-i);
  return c;
}
void bounded(const std::vector<Checkpoint>& cs, size_t pages, size_t slots) {
  size_t p=0,s=0;
  for (const auto& c:cs) if (c.last_reuse) { p+=c.pages.size(); ++s; }
  assert(p<=pages && s<=slots);
}
int main(int argc,char** argv) {
  const std::string mode=argv[1];
  if (mode=="scan") {
    for (bool protect:{false,true}) {
      std::vector<Checkpoint> cs;
      uint64_t clock=0;
      for (int i=0;i<8;++i) cs.push_back(ck(i,++clock));
      for (size_t i=0;i<3;++i) {
        cs[i].last_use=++clock;
        if (protect) protect_checkpoint(cs,i,clock,16,4);
      }
      for (int i=8;i<80;++i) {
        int victim=cache_eviction_index(cs,true);
        cs.erase(cs.begin()+victim);
        cs.push_back(ck(i,++clock));
        bounded(cs,16,4);
      }
      int hot=0;
      for (const auto& c:cs) hot+=c.ids[0]<3;
      assert(hot==(protect?3:0));
    }
  } else if (mode=="bounds_and_rotation") {
    std::vector<Checkpoint> cs;
    for (int i=0;i<8;++i) cs.push_back(ck(i,i+1));
    for (size_t i=0;i<8;++i) { protect_checkpoint(cs,i,10+i,12,4); bounded(cs,12,4); }
    for (int i=0;i<8;++i) assert(bool(cs[i].last_reuse)==(i>=5));
    // A new hit rotates protection. A save's recency alone does not earn it.
    cs[0].last_use=99;
    assert(!cs[0].last_reuse);
    protect_checkpoint(cs,0,100,12,4);
    assert(cs[0].last_reuse && !cs[5].last_reuse);
    trim_protected(cs,12,1); bounded(cs,12,1);
    trim_protected(cs,0,0); bounded(cs,0,0);
  } else if (mode=="oversize_and_shared") {
    std::vector<Checkpoint> cs={ck(0,1),ck(1,2),ck(2,3,20)};
    cs[1].pages=cs[0].pages; // shared physical pages are conservatively charged twice
    protect_checkpoint(cs,0,4,8,2);
    protect_checkpoint(cs,1,5,8,2);
    protect_checkpoint(cs,2,6,8,2); // oversized hit cannot flush both useful protected entries
    assert(cs[0].last_reuse && cs[1].last_reuse && !cs[2].last_reuse);
    auto before=cs;
    trim_protected(cs,4,2);
    assert(!cs[0].last_reuse && cs[1].last_reuse);
    for (size_t i=0;i<cs.size();++i) {
      assert(cs[i].pages==before[i].pages && cs[i].ids==before[i].ids);
      assert(cs[i].ring==before[i].ring && cs[i].slot==before[i].slot);
    }
  } else if (mode=="host_filter") {
    std::vector<Checkpoint> cs={ck(0,1),ck(1,2),ck(2,3)};
    cs[0].pages={1,2};
    protect_checkpoint(cs,1,4,8,2);
    assert(cache_eviction_index(cs)==0);
    assert(cache_eviction_index(cs,true)==2);
    cs[2].last_reuse=5;
    assert(cache_eviction_index(cs,true)==1); // all protected: eviction still makes progress
    assert(cache_eviction_index({},true)==-1);
  } else if (mode=="state_ranks_and_ring") {
    std::vector<Checkpoint> cs={ck(0,1),ck(1,2)};
    protect_checkpoint(cs,0,3,16,4);
    assert(state_eviction_index(cs,ck(9,10))==1); // equal semantic rank: protect actual reuse
    cs[1].ids={0,0,0}; // covered ancestor is still retired, even when protected
    assert(state_eviction_index(cs,ck(9,10))==0);
    cs[1].ring=false; // cannot cover the protected speculative ancestor
    assert(state_eviction_index(cs,ck(9,10))==1);
    assert(state_eviction_index(cs,ck(9,10),true)==-1); // optional interval cannot force eviction
  } else if (mode=="random") {
    std::mt19937 rng(23);
    std::vector<Checkpoint> cs;
    for (int i=0;i<32;++i) cs.push_back(ck(i,i+1,1+rng()%8));
    for (int t=33;t<20000;++t) {
      size_t i=rng()%cs.size();
      if (rng()%3) protect_checkpoint(cs,i,t,64,16);
      else cs[i]=ck(t,t,1+rng()%12);
      bounded(cs,64,16);
      int victim=cache_eviction_index(cs,true);
      assert(victim>=0);
      for (const auto& c:cs) assert(!eviction_before(c,cs[victim]));
    }
  } else return 1;
}
''', encoding="utf-8")
    binary = out / "policy"
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if os.environ.get("Q27_TEST_SANITIZERS") else []
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", *flags, "-I",
                    str(pathlib.Path(__file__).resolve().parents[1] / "src"), str(source), "-o", str(binary)],
                   check=True, capture_output=True)
    return binary


@pytest.mark.parametrize("case", ["scan", "bounds_and_rotation", "oversize_and_shared", "host_filter",
                                  "state_ranks_and_ring", "random"])
def test_protected_retention(policy_test, case):
    result = subprocess.run([str(policy_test), case], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()
