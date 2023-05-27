#ifndef CONSTANTS_H
#define CONSTANTS_H

class Constants{
public:
    static const int FP_SIZE = 12;
    static const int K_TAGS_PER_BUCKET = 4;
    static constexpr float REBALANCE_THRESHOLD = 0.5;
    static constexpr float DELETE_THRESHOLD = 0.1;
    static constexpr float FPR = 0.01;
    static const int L = 21;
};

#endif