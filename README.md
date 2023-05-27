LCF
============

Description
--------
LCF: A Learned Cuckoo Filter for Approximate Membership Queries over Variable-sized Sliding Windows on Data Streams.

LCF is a noval data structure that can provide satisfactory results for the approximate membership query on data streams, regardless of the user-defined query. LCF operates by adaptively maintaining cuckoo filters with the assistance of a well-trained oracle that learned the frequency feature of the data within the stream windows.

Dataset
--------
*  Internet traffic dataset: http://www.caida.org/data/passive/passive_dataset.xml
*  Search query dataset: https://jeffhuang.com/search_query_logs.html


Enviroment Request
--------------------
*  c++ 11+
*  openssl
*  libtorch(CPU or GPU based on your own CUDA version): https://pytorch.org/get-started/locally/


Usage:
Parameter can be set in src/Constant.cpp
```bash
$ mkdir build
$ cd build/
$ cmake .. && make -j8
$ ./test
```
