#ifndef CUCKOO_FILTER_SINGLE_TABLE_H_
#define CUCKOO_FILTER_SINGLE_TABLE_H_

#include <assert.h>
#include <iostream>

#include <sstream>

#include "bitsutil.h"
#include "debug.h"
#include "printutil.h"
#include "constants.h"

using namespace std;

namespace cuckoofilter {

// the most naive table implementation: one huge bit array
template <size_t bits_per_tag>
class SingleTable {
public:
  size_t kTagsPerBucket = Constants::K_TAGS_PER_BUCKET;
  // static const size_t kBytesPerBucket =
  //     (bits_per_tag * kTagsPerBucket + 7) >> 3;
  
  // 每个tag 纵向排列 方便删除每个bucket最后的tag
  static const size_t kBytesPerBucket =
      (bits_per_tag + 7) >> 3;
  static const uint32_t kTagMask = (1ULL << bits_per_tag) - 1;
  // NOTE: accomodate extra buckets if necessary to avoid overrun
  // as we always read a uint64
  static const size_t kPaddingBuckets =
    ((((kBytesPerBucket + 7) / 8) * 8) - 1) / kBytesPerBucket;

  struct Bucket {
    char bits_[kBytesPerBucket];
  } __attribute__((__packed__));

  // using a pointer adds one more indirection
  Bucket *buckets_;
  size_t num_buckets_;

 public:
  explicit SingleTable(const size_t num) : num_buckets_(num) {
    buckets_ = new Bucket[num_buckets_*kTagsPerBucket + kPaddingBuckets];
    memset(buckets_, 0, kBytesPerBucket * (num_buckets_*kTagsPerBucket + kPaddingBuckets));
  }

  ~SingleTable() { 
    delete[] buckets_;
  }

  size_t NumBuckets() const {
    return num_buckets_;
  }

  size_t SizeInBytes() const { 
    return kBytesPerBucket * num_buckets_ * kTagsPerBucket; 
  }

  size_t SizeInTags() const { 
    return kTagsPerBucket * num_buckets_; 
  }

  std::string Info() const {
    std::stringstream ss;
    ss << "SingleHashtable with tag size: " << bits_per_tag << " bits \n";
    ss << "\t\tAssociativity: " << kTagsPerBucket << "\n";
    ss << "\t\tTotal # of rows: " << num_buckets_ << "\n";
    ss << "\t\tTotal # slots: " << SizeInTags() << "\n";
    return ss.str();
  }

  // read tag from pos(i,j)
  inline uint32_t ReadTag(const size_t i, const size_t j) const {
    const char *p = buckets_[i].bits_;
    uint32_t tag;
    /* following code only works for little-endian */
    if (bits_per_tag == 2) {
      tag = *((uint8_t *)p) >> (j * 2);
    } else if (bits_per_tag == 4) {
      p += (j >> 1);
      tag = *((uint8_t *)p) >> ((j & 1) << 2);
    } else if (bits_per_tag == 8) {
      p += j;
      tag = *((uint8_t *)p);
    } else if (bits_per_tag == 12) {
      p += j + (j >> 1);
      tag = *((uint16_t *)p) >> ((j & 1) << 2);
    } else if (bits_per_tag == 16) {
      p += (j << 1);
      tag = *((uint16_t *)p);
    } else if (bits_per_tag == 32) {
      tag = ((uint32_t *)p)[j];
    }
    return tag & kTagMask;
  }

  // write tag to pos(i,j)
  inline void WriteTag(const size_t i, const size_t j, const uint32_t t) {
    char *p = buckets_[i].bits_;
    uint32_t tag = t & kTagMask;
    /* following code only works for little-endian */
    if (bits_per_tag == 2) {
      *((uint8_t *)p) |= tag << (2 * j);
    } else if (bits_per_tag == 4) {
      p += (j >> 1);
      if ((j & 1) == 0) {
        *((uint8_t *)p) &= 0xf0;
        *((uint8_t *)p) |= tag;
      } else {
        *((uint8_t *)p) &= 0x0f;
        *((uint8_t *)p) |= (tag << 4);
      }
    } else if (bits_per_tag == 8) {
      ((uint8_t *)p)[j] = tag;
    } else if (bits_per_tag == 12) {
      p += (j + (j >> 1));
      if ((j & 1) == 0) {
        ((uint16_t *)p)[0] &= 0xf000;
        ((uint16_t *)p)[0] |= tag;
      } else {
        ((uint16_t *)p)[0] &= 0x000f;
        ((uint16_t *)p)[0] |= (tag << 4);
      }
    } else if (bits_per_tag == 16) {
      ((uint16_t *)p)[j] = tag;
    } else if (bits_per_tag == 32) {
      ((uint32_t *)p)[j] = tag;
    }
  }

  inline bool FindTagInBuckets(const size_t i1, const size_t i2,
                               const uint32_t tag) const {
    // const char *p1 = buckets_[i1].bits_;
    // const char *p2 = buckets_[i2].bits_;

    // uint64_t v1 = *((uint64_t *)p1);
    // uint64_t v2 = *((uint64_t *)p2);

    // caution: unaligned access & assuming little endian
    // if (bits_per_tag == 4 && kTagsPerBucket == 4) {
    //   return hasvalue4(v1, tag) || hasvalue4(v2, tag);
    // } else if (bits_per_tag == 8 && kTagsPerBucket == 4) {
    //   return hasvalue8(v1, tag) || hasvalue8(v2, tag);
    // } else if (bits_per_tag == 12 && kTagsPerBucket == 4) {
    //   return hasvalue12(v1, tag) || hasvalue12(v2, tag);
    // } else if (bits_per_tag == 16 && kTagsPerBucket == 4) {
    //   return hasvalue16(v1, tag) || hasvalue16(v2, tag);
    // } else {
    //   for (size_t j = 0; j < kTagsPerBucket; j++) {
    //     if ((ReadTag(i1+j*num_buckets_, 0) == tag) || (ReadTag(i2+j*num_buckets_, 0) == tag)) {
    //       return true;
    //     }
    //   }
    //   return false;
    // }

    for (size_t j = 0; j < kTagsPerBucket; j++) {
      if ((ReadTag(i1+j*num_buckets_, 0) == tag) || (ReadTag(i2+j*num_buckets_, 0) == tag)) {
        return true;
      }
    }
    return false;
  }

  inline bool FindTagInBucket(const size_t i, const uint32_t tag) const {
    // caution: unaligned access & assuming little endian
    // if (bits_per_tag == 4 && kTagsPerBucket == 4) {
    //   const char *p = buckets_[i].bits_;
    //   uint64_t v = *(uint64_t *)p;  // uint16_t may suffice
    //   return hasvalue4(v, tag);
    // } else if (bits_per_tag == 8 && kTagsPerBucket == 4) {
    //   const char *p = buckets_[i].bits_;
    //   uint64_t v = *(uint64_t *)p;  // uint32_t may suffice
    //   return hasvalue8(v, tag);
    // } else if (bits_per_tag == 12 && kTagsPerBucket == 4) {
    //   const char *p = buckets_[i].bits_;
    //   uint64_t v = *(uint64_t *)p;
    //   return hasvalue12(v, tag);
    // } else if (bits_per_tag == 16 && kTagsPerBucket == 4) {
    //   const char *p = buckets_[i].bits_;
    //   uint64_t v = *(uint64_t *)p;
    //   return hasvalue16(v, tag);
    // } else {
    //   for (size_t j = 0; j < kTagsPerBucket; j++) {
    //     if (ReadTag(i, j) == tag) {
    //       return true;
    //     }
    //   }
    //   return false;
    // }
    for (size_t j = 0; j < kTagsPerBucket; j++) {
      if (ReadTag(i+j*num_buckets_, 0) == tag) {
        return true;
      }
    }
    return false;
  }

  inline bool DeleteTagFromBucket(const size_t i, const uint32_t tag) {
    for (size_t j = 0; j < kTagsPerBucket; j++) {
      if (ReadTag(i+j*num_buckets_, 0) == tag) {
        assert(FindTagInBucket(i, tag) == true);
        WriteTag(i+j*num_buckets_, 0, 0);
        return true;
      }
    }
    return false;
  }

  inline bool InsertTagToBucket(const size_t i, const uint32_t tag,
                                const bool kickout, uint32_t &oldtag) {
    for (size_t j = 0; j < kTagsPerBucket; j++) {
      if (ReadTag(i+j*num_buckets_, 0) == 0) {
        WriteTag(i+j*num_buckets_, 0, tag);
        return true;
      }
    }
    if (kickout) {
      size_t r = rand() % kTagsPerBucket;
      oldtag = ReadTag(i+r*num_buckets_, 0);
      WriteTag(i+r*num_buckets_, 0, tag);
    }
    return false;
  }

  inline size_t NumTagsInBucket(const size_t i) const {
    size_t num = 0;
    for (size_t j = 0; j < kTagsPerBucket; j++) {
      if (ReadTag(i+j*num_buckets_, 0) != 0) {
        num++;
      }
    }
    return num;
  }

  inline void PrintTable() {
    cout << "kBytesPerBucket : " << kBytesPerBucket << endl;
    int len = num_buckets_*kTagsPerBucket + kPaddingBuckets;
    cout << "table len : " << len << endl;
    for(int i = 0; i < len; ++i){
      cout << ReadTag(i, 0) << endl;
    }
  }

  void ResizeTable(size_t newTagsPerBucket){
    Bucket* new_bucket = new Bucket[num_buckets_*newTagsPerBucket + kPaddingBuckets];
    memcpy(new_bucket, buckets_, sizeof(Bucket)*(num_buckets_*newTagsPerBucket + kPaddingBuckets));
    delete[] buckets_;
    buckets_ = new_bucket;
  }
};
}  // namespace cuckoofilter
#endif  // CUCKOO_FILTER_SINGLE_TABLE_H_
