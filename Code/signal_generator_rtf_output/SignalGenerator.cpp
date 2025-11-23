// SignalGenerator.cpp
#include "SignalGenerator.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

// =================== Utilities ===================

int SignalGenerator::mod(int a, int b) {
    int ret = a % b;
    if (ret < 0) ret += b;
    return ret;
}

double SignalGenerator::sinc(double x) {
    if (x == 0.0) return 1.0;
    return std::sin(x) / x;
}

double SignalGenerator::sim_microphone(double x, double y, double z,
                                       const std::vector<double>& angle,
                                       char mtype) {
    // Cardioid family + hypercardioid
    if (mtype == 'b' || mtype == 'c' || mtype == 's' || mtype == 'h') {
        double alpha = 1.0;
        switch (mtype) {
            case 'b': alpha = 0.0;  break;  // bidirectional (figure-8)
            case 'h': alpha = 0.25; break;  // hypercardioid
            case 'c': alpha = 0.5;  break;  // cardioid
            case 's': alpha = 0.75; break;  // subcardioid
        }
        double r = std::sqrt(x*x + y*y + z*z);
        if (r == 0.0) return 1.0;

        double vartheta = std::acos(z / r);
        double varphi   = std::atan2(y, x);
        double strength =
            std::sin(M_PI/2 - angle[1]) * std::sin(vartheta) * std::cos(angle[0] - varphi) +
            std::cos(M_PI/2 - angle[1]) * std::cos(vartheta);
        return alpha + (1.0 - alpha) * strength;
    }
    // Omnidirectional default
    return 1.0;
}

bool SignalGenerator::IsSrcPosConst(const std::vector<std::vector<double>>& ss,
                                    size_t t, size_t offset) {
    if (t <= offset) return false;
    return ss[t - offset][0] == ss[t - offset - 1][0] &&
           ss[t - offset][1] == ss[t - offset - 1][1] &&
           ss[t - offset][2] == ss[t - offset - 1][2];
}

bool SignalGenerator::IsRcvPosConst(const std::vector<std::vector<std::vector<double>>>& rr,
                                    size_t mic_idx, size_t t) {
    if (t == 0) return false;
    return rr[t][0][mic_idx] == rr[t - 1][0][mic_idx] &&
           rr[t][1][mic_idx] == rr[t - 1][1][mic_idx] &&
           rr[t][2][mic_idx] == rr[t - 1][2][mic_idx];
}

void SignalGenerator::copy_previous_rir(std::vector<double>& imp,
                                        int row_idx, int nsamples) {
    if (row_idx == 0) {
        for (int i = 0; i < nsamples; ++i)
            imp[i * nsamples] = imp[i * nsamples + (nsamples - 1)];
    } else {
        for (int i = 0; i < nsamples; ++i)
            imp[i * nsamples + row_idx] = imp[i * nsamples + (row_idx - 1)];
    }
}

void SignalGenerator::hpf_imp(std::vector<double>& imp, int row_idx, int nsamples, const HPF& hpf) {
    double Y[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < nsamples; ++i) {
        double X0 = imp[row_idx + nsamples * i];
        Y[2] = Y[1];
        Y[1] = Y[0];
        Y[0] = hpf.B1 * Y[1] + hpf.B2 * Y[2] + X0;
        imp[row_idx + nsamples * i] = Y[0] + hpf.A1 * Y[1] + hpf.R1 * Y[2];
    }
}

// =================== Core generator ===================

SignalGenerator::Result SignalGenerator::generate(
    const std::vector<double>& input_signal,
    double c,
    double fs,
    const std::vector<std::vector<std::vector<double>>>& r_path,  // [T][3][M]
    const std::vector<std::vector<double>>& s_path,                // [T][3]
    const std::vector<double>& L,
    const std::vector<double>& beta_or_tr,
    int nsamples,
    const std::string& mtype_str,
    int order,
    int dim,
    const std::vector<std::vector<double>>& orientation,
    bool hp_filter
) {
    // ---- Validate ----
    const size_t N = input_signal.size();
    if (N != r_path.size() || N != s_path.size())
        throw std::invalid_argument("Signal length must match r_path and s_path time dimension.");
    if (L.size() != 3)
        throw std::invalid_argument("Room dimensions (L) must be 3.");
    const int no_mics = static_cast<int>(r_path[0][0].size());
    if (!(beta_or_tr.size() == 6 || beta_or_tr.size() == 1))
        throw std::invalid_argument("beta_or_tr must be length 6 (per wall) or 1 (TR).");
    (void)dim; // not used here but kept for API compatibility

    // ---- Beta / TR ----
    std::vector<double> beta(6, 0.0);
    double beta_hat = 0.0;
    double TR = 0.0;

    if (beta_or_tr.size() == 1) {
        TR = beta_or_tr[0];
        const double V = L[0]*L[1]*L[2];
        const double S = 2.0*(L[0]*L[2] + L[1]*L[2] + L[0]*L[1]);
        const double alpha = 24.0 * V * std::log(10.0) / (c * S * TR);
        if (alpha > 1.0)
            throw std::runtime_error("Invalid TR/room: absorption exceeds 1.");
        beta_hat = std::sqrt(1.0 - alpha);
        std::fill(beta.begin(), beta.end(), beta_hat);
    } else {
        beta = beta_or_tr;
    }

    // ---- nsamples if not provided ----
    if (nsamples <= 0) {
        const double V = L[0]*L[1]*L[2];
        const double S = 2.0*(L[0]*L[2] + L[1]*L[2] + L[0]*L[1]);
        const double alphas =
            ((1 - std::pow(beta[0],2)) + (1 - std::pow(beta[1],2))) * L[0] * L[2] +
            ((1 - std::pow(beta[2],2)) + (1 - std::pow(beta[3],2))) * L[1] * L[2] +
            ((1 - std::pow(beta[4],2)) + (1 - std::pow(beta[5],2))) * L[0] * L[1];
        TR = 24.0 * std::log(10.0) * V / (c * alphas);
        TR = std::max(TR, 0.128);  // clamp minimum RT
        nsamples = static_cast<int>(TR * fs);
    }

    // ---- Mic types ----
    std::vector<char> mtypes(no_mics, 'o');
    if (!mtype_str.empty()) {
        if (mtype_str.size() == 1) {
            std::fill(mtypes.begin(), mtypes.end(), mtype_str[0]);
        } else if (static_cast<int>(mtype_str.size()) == no_mics) {
            for (int i = 0; i < no_mics; ++i) mtypes[i] = mtype_str[i];
        } else {
            throw std::invalid_argument("mtype must be length 1 or equal to number of microphones.");
        }
    }

    // ---- Orientation ----
    std::vector<std::vector<double>> angles(no_mics, std::vector<double>(2, 0.0));
    if (!orientation.empty()) {
        if (orientation.size() == 1 && orientation[0].size() == 1) {
            for (int i = 0; i < no_mics; ++i) angles[i][0] = orientation[0][0];
        } else if (orientation.size() == 1 && orientation[0].size() == 2) {
            for (int i = 0; i < no_mics; ++i) { angles[i][0] = orientation[0][0]; angles[i][1] = orientation[0][1]; }
        } else if (orientation.size() == static_cast<size_t>(no_mics) && orientation[0].size() == 1) {
            for (int i = 0; i < no_mics; ++i) angles[i][0] = orientation[i][0];
        } else if (orientation.size() == static_cast<size_t>(no_mics) && orientation[0].size() == 2) {
            angles = orientation;
        } else {
            throw std::invalid_argument("Invalid orientation shape.");
        }
    }

    // ---- Geometry scaling ----
    const double cTs = c / fs;
    std::vector<double> L_scaled(3);
    for (int i = 0; i < 3; ++i) L_scaled[i] = L[i] / cTs;

    std::vector<int> nimg(3);
    for (int i = 0; i < 3; ++i) nimg[i] = static_cast<int>(std::ceil(nsamples / (2.0 * L_scaled[i])));

    const int Tw = 2 * static_cast<int>(std::round(0.004 * fs));
    std::vector<double> hanning_window(Tw + 1);
    for (int i = 0; i <= Tw; ++i)
        hanning_window[i] = 0.5 * (1.0 + std::cos(2.0 * M_PI * (i + Tw / 2.0) / Tw));

    // ---- STFT frame mapping (KEEP IN SYNC WITH PYTHON) ----
    const int STFT_WIN = 512;   // <-- your window length
    const int STFT_HOP = 128;   // <-- your hop size

    const int N_int = static_cast<int>(N);
    const int L_frames = (N_int >= STFT_WIN) ? ((N_int - STFT_WIN) / STFT_HOP + 1) : 0;
    // Optional: frame center sample indices (not returned)
    // std::vector<int> frame_samples(L_frames);
    // for (int f = 0; f < L_frames; ++f) frame_samples[f] = f*STFT_HOP + STFT_WIN/2;

    // ---- Outputs ----
    std::vector<std::vector<double>> output(no_mics, std::vector<double>(N, 0.0));
    std::vector<std::vector<std::vector<double>>> all_rirs(
        no_mics,
        std::vector<std::vector<double>>(L_frames, std::vector<double>(nsamples, 0.0))
    );

    // ---- High-pass filter ----
    HPF hpf;
    hpf.W  = 2.0 * M_PI * 100.0 / fs;
    hpf.R1 = std::exp(-hpf.W);
    hpf.B1 = 2.0 * hpf.R1 * std::cos(hpf.W);
    hpf.B2 = -std::pow(hpf.R1, 2);
    hpf.A1 = -(1.0 + hpf.R1);

    // ---- Scratch ----
    std::vector<double> LPI(Tw + 1);
    // flattened [nsamples x nsamples] ring buffer for time-varying RIR rows
    std::vector<double> imp(nsamples * nsamples, 0.0);
    std::vector<double> r(3), s(3), hu(6), refl(3);

    // =================== Main loops ===================
    for (int mic_idx = 0; mic_idx < no_mics; ++mic_idx) {
        const auto& angle = angles[mic_idx];
        std::fill(imp.begin(), imp.end(), 0.0);

        for (size_t t = 0; t < N; ++t) {
            const int row_idx_1 = static_cast<int>(t % nsamples);

            // Receiver position (normalized)
            for (int i = 0; i < 3; ++i)
                r[i] = r_path[t][i][mic_idx] / cTs;

            // Invariance checks
            const bool bSrcInvariant_1 = (t > 0) && IsSrcPosConst(s_path, t, 0);
            const bool bRcvInvariant_1 = (t > 0) && IsRcvPosConst(r_path, mic_idx, t);

            int no_rows_to_update = 0;
            if (!(bSrcInvariant_1 && bRcvInvariant_1)) {
                no_rows_to_update = (bRcvInvariant_1 || t == 0)
                    ? 1
                    : std::min<int>(static_cast<int>(t), nsamples);

                for (int row_counter = 0; row_counter < no_rows_to_update; ++row_counter) {
                    const int row_idx_2 = mod(row_idx_1 - row_counter, nsamples);
                    const bool bSrcInvariant_2 = (row_counter > 0) && IsSrcPosConst(s_path, t, row_counter);

                    if (!bSrcInvariant_2) {
                        for (int i = 0; i < 3; ++i)
                            s[i] = s_path[t - row_counter][i] / cTs;

                        // clear RIR row
                        for (int i = 0; i < nsamples; ++i)
                            imp[row_idx_2 + nsamples * i] = 0.0;

                        for (int mx = -nimg[0]; mx <= nimg[0]; ++mx) {
                            hu[0] = 2 * mx * L_scaled[0];
                            for (int my = -nimg[1]; my <= nimg[1]; ++my) {
                                hu[1] = 2 * my * L_scaled[1];
                                for (int mz = -nimg[2]; mz <= nimg[2]; ++mz) {
                                    hu[2] = 2 * mz * L_scaled[2];
                                    for (int q = 0; q <= 1; ++q) {
                                        hu[3] = (1 - 2*q) * s[0] - r[0] + hu[0];
                                        refl[0] = std::pow(beta[0], std::abs(mx - q)) * std::pow(beta[1], std::abs(mx));
                                        for (int j = 0; j <= 1; ++j) {
                                            hu[4] = (1 - 2*j) * s[1] - r[1] + hu[1];
                                            refl[1] = std::pow(beta[2], std::abs(my - j)) * std::pow(beta[3], std::abs(my));
                                            for (int k = 0; k <= 1; ++k) {
                                                hu[5] = (1 - 2*k) * s[2] - r[2] + hu[2];
                                                refl[2] = std::pow(beta[4], std::abs(mz - k)) * std::pow(beta[5], std::abs(mz));

                                                if ( (std::abs(2*mx - q) + std::abs(2*my - j) + std::abs(2*mz - k) <= order) || (order == -1) ) {
                                                    const double dist  = std::sqrt(hu[3]*hu[3] + hu[4]*hu[4] + hu[5]*hu[5]);
                                                    const int    fdist = static_cast<int>(std::floor(dist));
                                                    if (fdist < nsamples) {
                                                        for (int idx = 0; idx <= Tw; ++idx)
                                                            LPI[idx] = hanning_window[idx] *
                                                                       sinc(M_PI * (idx - (dist - fdist) - Tw / 2));

                                                        for (int idx = 0; idx <= Tw; ++idx) {
                                                            const int pos = fdist - Tw/2 + idx;
                                                            if (pos >= 0 && pos < nsamples) {
                                                                const double strength =
                                                                    sim_microphone(hu[3], hu[4], hu[5], angle, mtypes[mic_idx]) *
                                                                    refl[0] * refl[1] * refl[2] /
                                                                    (4 * M_PI * dist * cTs);
                                                                imp[row_idx_2 + nsamples * pos] += strength * LPI[idx];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (hp_filter) hpf_imp(imp, row_idx_2, nsamples, hpf);
                    } else {
                        copy_previous_rir(imp, row_idx_2, nsamples);
                    }
                }
            } else {
                copy_previous_rir(imp, row_idx_1, nsamples);
            }

            // ---- Snapshot at STFT frame centers (per-frame RIR) ----
            if (L_frames > 0) {
                const int first_center = STFT_WIN / 2;
                if (static_cast<int>(t) >= first_center) {
                    int dt = static_cast<int>(t) - first_center;
                    if (dt % STFT_HOP == 0) {
                        int f = dt / STFT_HOP; // frame index
                        if (0 <= f && f < L_frames) {
                            for (int k = 0; k < nsamples; ++k)
                                all_rirs[mic_idx][f][k] = imp[row_idx_1 + nsamples * k];
                        }
                    }
                }
            }

            // ---- Final convolution (apply current imp row) ----
            for (int k = 0; k < nsamples; ++k) {
                if (t >= static_cast<size_t>(k)) {
                    const int tmp_imp_idx = mod(row_idx_1 - k, nsamples);
                    output[mic_idx][t] += imp[tmp_imp_idx + nsamples * k] * input_signal[t - k];
                }
            }
        } // t
    } // mic

    return Result{ output, beta_hat, all_rirs };
}
