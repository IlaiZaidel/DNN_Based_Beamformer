// SignalGenerator.h
#pragma once
#include <vector>

class SignalGenerator {
public:
    struct Result {
        std::vector<std::vector<double>> output;                 // [2][T]
        double beta_hat;                                         // always 0 for HRTF
        std::vector<std::vector<std::vector<double>>> all_rirs;  // [2][L_frames][K]
    };

    Result generate_hrtf(
        const std::vector<double>& input_signal,                  // [T]
        const std::vector<std::vector<double>>& s_path,           // [T][3]
        const std::vector<std::vector<double>>& r_head_path,      // [T][3]
        const std::vector<std::vector<std::vector<double>>>& hrirs,// [N][2][K]
        const std::vector<std::vector<double>>& azelevs,          // [N][2]
        int stft_win = 512,
        int stft_hop = 128,
        int update_every = 128
    );
};
