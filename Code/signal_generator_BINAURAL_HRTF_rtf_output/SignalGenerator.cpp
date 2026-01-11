// SignalGenerator.cpp  (HRTF-only)
// Implements ONLY: SignalGenerator::generate_hrtf(...)
// No Habets room RIR generator, no HPF, no mic directivity, no extra class utilities.

#include "SignalGenerator.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// -------- helpers (file-local, NOT class methods) --------

// wrap-around aware azimuth diff in degrees
static inline double wrap_az_diff_deg(double a, double b) {
    double d = std::fabs(a - b);
    return std::min(d, 360.0 - d);
}

// Cartesian -> azimuth/elevation in degrees
static inline void cart2azelev_deg(double dx, double dy, double dz,
                                   double& az, double& el) {
    az = std::atan2(dy, dx) * 180.0 / M_PI;   // [-180,180]
    if (az < 0.0) az += 360.0;                // [0,360)
    el = std::atan2(dz, std::sqrt(dx*dx + dy*dy)) * 180.0 / M_PI; // [-90,90]
}

// nearest neighbor search in (az,elev) with az wrap-around
static inline int nearest_dir_idx(double az, double el,
                                  const std::vector<std::vector<double>>& azelevs) {
    double best = 1e30;
    int best_i = 0;
    for (int i = 0; i < (int)azelevs.size(); ++i) {
        const double az_i = azelevs[i][0];
        const double el_i = azelevs[i][1];
        const double da = wrap_az_diff_deg(az, az_i);
        const double de = (el - el_i);
        const double d2 = da*da + de*de;
        if (d2 < best) { best = d2; best_i = i; }
    }
    return best_i;
}

// -------- main implementation --------

SignalGenerator::Result SignalGenerator::generate_hrtf(
    const std::vector<double>& input_signal,                  // [T]
    const std::vector<std::vector<double>>& s_path,           // [T][3]
    const std::vector<std::vector<double>>& r_head_path,      // [T][3]
    const std::vector<std::vector<std::vector<double>>>& hrirs,// [N][2][K]
    const std::vector<std::vector<double>>& azelevs,          // [N][2] degrees
    int stft_win,
    int stft_hop,
    int update_every
) {
    const size_t T = input_signal.size();
    if (T == 0) throw std::invalid_argument("generate_hrtf: input_signal is empty.");
    if (s_path.size() != T || r_head_path.size() != T)
        throw std::invalid_argument("generate_hrtf: s_path and r_head_path must match input_signal length.");
    for (size_t t = 0; t < T; ++t) {
        if (s_path[t].size() != 3 || r_head_path[t].size() != 3)
            throw std::invalid_argument("generate_hrtf: s_path/r_head_path must be [T][3].");
    }

    if (hrirs.empty() || azelevs.empty())
        throw std::invalid_argument("generate_hrtf: hrirs/azelevs must be non-empty.");
    if (hrirs.size() != azelevs.size())
        throw std::invalid_argument("generate_hrtf: hrirs and azelevs must have same N_dirs.");
    if (hrirs[0].size() != 2)
        throw std::invalid_argument("generate_hrtf: hrirs must be [N][2][K].");
    if (azelevs[0].size() != 2)
        throw std::invalid_argument("generate_hrtf: azelevs must be [N][2].");

    const int K = (int)hrirs[0][0].size();
    if (K <= 0)
        throw std::invalid_argument("generate_hrtf: HRIR length K must be > 0.");
    for (size_t i = 0; i < hrirs.size(); ++i) {
        if (hrirs[i].size() != 2) throw std::invalid_argument("generate_hrtf: hrirs must be [N][2][K].");
        if ((int)hrirs[i][0].size() != K || (int)hrirs[i][1].size() != K)
            throw std::invalid_argument("generate_hrtf: all hrirs must share same K.");
        if (azelevs[i].size() != 2) throw std::invalid_argument("generate_hrtf: azelevs must be [N][2].");
    }

    if (stft_win <= 0 || stft_hop <= 0)
        throw std::invalid_argument("generate_hrtf: stft_win and stft_hop must be > 0.");
    if (update_every <= 0)
        throw std::invalid_argument("generate_hrtf: update_every must be > 0.");

    // frame count (same formula as before)
    const int T_int = (int)T;
    const int L_frames = (T_int >= stft_win) ? ((T_int - stft_win) / stft_hop + 1) : 0;

    // output [2][T]
    std::vector<std::vector<double>> output(2, std::vector<double>(T, 0.0));

    // snapshots [2][L_frames][K]
    std::vector<std::vector<std::vector<double>>> all_rirs(
        2,
        std::vector<std::vector<double>>(L_frames, std::vector<double>(K, 0.0))
    );

    // current HRIR taps (piecewise constant)
    std::vector<double> hL(K, 0.0), hR(K, 0.0);
    int last_idx = -1;

    auto load_hrir_for_time = [&](size_t t) {
        const double dx = s_path[t][0] - r_head_path[t][0];
        const double dy = s_path[t][1] - r_head_path[t][1];
        const double dz = s_path[t][2] - r_head_path[t][2];

        double az = 0.0, el = 0.0;
        cart2azelev_deg(dx, dy, dz, az, el);

        const int idx = nearest_dir_idx(az, el, azelevs);
        last_idx = idx;

        for (int k = 0; k < K; ++k) {
            hL[k] = hrirs[idx][0][k];
            hR[k] = hrirs[idx][1][k];
        }
    };

    // init
    load_hrir_for_time(0);

    const int first_center = stft_win / 2;

    for (size_t t = 0; t < T; ++t) {
        // update cadence (recommend: update_every = stft_hop)
        if (t > 0 && (t % (size_t)update_every) == 0) {
            load_hrir_for_time(t);
        }

        // snapshot at frame centers
        if (L_frames > 0 && (int)t >= first_center) {
            const int dt = (int)t - first_center;
            if (dt % stft_hop == 0) {
                const int f = dt / stft_hop;
                if (0 <= f && f < L_frames && last_idx >= 0) {
                    for (int k = 0; k < K; ++k) {
                        all_rirs[0][f][k] = hL[k];
                        all_rirs[1][f][k] = hR[k];
                    }
                }
            }
        }

        // causal FIR
        for (int k = 0; k < K; ++k) {
            if (t >= (size_t)k) {
                const double x = input_signal[t - (size_t)k];
                output[0][t] += hL[k] * x;
                output[1][t] += hR[k] * x;
            }
        }
    }

    return Result{ output, 0.0, all_rirs };
}
