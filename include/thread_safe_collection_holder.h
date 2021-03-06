#pragma once

#include <queue>
#include <unordered_map>
#include <memory>
#include <vector>
#include <utility>

#include "boost/thread/locks.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

template<typename K, typename T>
class ThreadSafeCollectionHolder : boost::noncopyable {
 public:
  ThreadSafeCollectionHolder()
      : lock_(), object_(std::unordered_map<K, std::shared_ptr<T>>()) { }

  static ThreadSafeCollectionHolder<K, T>& singleton() {
    // Mayers singleton is thread safe in C++11
    // http://stackoverflow.com/questions/1661529/is-meyers-implementation-of-singleton-pattern-thread-safe
    static ThreadSafeCollectionHolder<K, T> holder;
    return holder;
  }

  ~ThreadSafeCollectionHolder() { }

  std::shared_ptr<T> get(const K& key) const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return get_locked(key);
  }

  bool has_key(const K& key) const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return object_.find(key) != object_.end();
  }

  void erase(const K& key) {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto iter = object_.find(key);
    if (iter != object_.end()) {
      object_.erase(iter);
    }
  }

  void clear() {
    boost::lock_guard<boost::mutex> guard(lock_);
    object_.clear();
  }

  std::shared_ptr<T> get_copy(const K& key) const {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto value = get_locked(key);
    return value != nullptr ? std::make_shared<T>(*value) : std::shared_ptr<T>();
  }

  void set(const K& key, const std::shared_ptr<T>& object) {
    boost::lock_guard<boost::mutex> guard(lock_);
    auto iter = object_.find(key);
    if (iter != object_.end()) {
      iter->second = object;
    } else {
      object_.insert(std::pair<K, std::shared_ptr<T> >(key, object));
    }
  }

  std::vector<K> keys() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    std::vector<K> retval;
    for (auto iter = object_.begin(); iter != object_.end(); ++iter) {
      retval.push_back(iter->first);
    }

    return retval;
  }

  size_t size() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return object_.size();
  }

  bool empty() const {
    boost::lock_guard<boost::mutex> guard(lock_);
    return object_.empty();
  }

 private:
  mutable boost::mutex lock_;
  std::unordered_map<K, std::shared_ptr<T> > object_;

  // Use this instead of get() when the lock is already acquired.
  std::shared_ptr<T> get_locked(const K& key) const {
    auto iter = object_.find(key);
    return (iter != object_.end()) ? iter->second : std::shared_ptr<T>();
  }
};
