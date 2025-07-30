// SignalGenerator.h
#pragma once

#include <vector>
#include <string>

class SignalGenerator {
public:
    struct Result {
        std::vector<std::vector<double>> output;
        double beta_hat;
    };

    Result generate(
        const std::vector<double>& input_signal,
        double c,
        double fs,
        const std::vector<std::vector<std::vector<double>>>& r_path,
        const std::vector<std::vector<double>>& s_path,
        const std::vector<double>& L,
        const std::vector<double>& beta_or_tr,
        int nsamples = -1,
        const std::string& mtype = "o",
        int order = -1,
        int dim = 3,
        const std::vector<std::vector<double>>& orientation = {},
        bool hp_filter = true
    );

public:
    struct HPF {
        double W;
        double R1;
        double B1;
        double B2;
        double A1;
    };

    static int mod(int a, int b);
    static double sinc(double x);
    static bool IsSrcPosConst(const std::vector<std::vector<double>>& ss, size_t t, size_t offset);
    static bool IsRcvPosConst(const std::vector<std::vector<std::vector<double>>>& rr, size_t mic_idx, size_t t);
    static double sim_microphone(double x, double y, double z, const std::vector<double>& angle, char mtype);
    static void hpf_imp(std::vector<double>& imp, int row_idx, int nsamples, const HPF& hpf);
    static void copy_previous_rir(std::vector<double>& imp, int row_idx, int nsamples);
};
