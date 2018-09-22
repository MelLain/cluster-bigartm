#pragma once

#include <string>
#include <vector>

#include "common.h"

template<class T, class V>
bool repeated_field_contains(const T& field, V value) {
  for (int i = 0; i < field.size(); ++i) {
    if (field.Get(i) == value) {
      return true;
    }
  }

  return false;
}

template<class T, class V>
int repeated_field_index_of(const T& field, V value) {
  for (int i = 0; i < field.size(); ++i) {
    if (field.Get(i) == value) {
      return i;
    }
  }

  return -1;
}

template<class T, class V>
void repeated_field_append(T* field, int index, V value) {
  V new_value = field->Get(index) + value;
  field->Set(index, new_value);
}

template<class T>
std::vector<bool> is_member(const T& elements, const T& set) {
  std::vector<bool> retval;
  retval.assign(elements.size(), false);

  if (elements.size() > 0) {
    for (int j = 0; j < set.size(); ++j) {
      for (int i = 0; i < elements.size(); ++i) {
        if (set.Get(j) == elements.Get(i)) {
          retval[i] = true;
          break;
        }
      }
    }
  }

  return retval;
}

template<class T, class V>
bool is_member(const V& value, const T& set) {
  for (int i = 0; i < set.size(); ++i) {
    if (set.Get(i) == value) {
      return true;
    }
  }

  return false;
}

template<class T>
bool repeated_field_equals(const T& f1, const T& f2) {
  if (f1.size() != f2.size()) {
    return false;
  }
  for (int i = 0; i < f1.size(); i++) {
    if (f1.Get(i) != f2.Get(i)) {
      return false;
    }
  }
  return true;
}
