// SignalGenerator.cpp
#include "SignalGenerator.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;

// Utility: Safe modulo
int SignalGenerator::mod(int a, int b) {
    int ret = a % b;
    if (ret < 0) ret += b;
    return ret;
}

// Utility: Sinc function
double SignalGenerator::sinc(double x) {
    if (x == 0.0) return 1.0;
    return sin(x) / x;
}

// Utility: Microphone directivity simulation
double SignalGenerator::sim_microphone(double x, double y, double z, const vector<double>& angle, char mtype) {
    if (mtype == 'b' || mtype == 'c' || mtype == 's' || mtype == 'h') {
        double alpha = 1.0;
        switch (mtype) {
            case 'b': alpha = 0.0; break;
            case 'h': alpha = 0.25; break;
            case 'c': alpha = 0.5; break;
            case 's': alpha = 0.75; break;
        }

        double r = sqrt(x * x + y * y + z * z);
        if (r == 0) return 1.0;

        double vartheta = acos(z / r);
        double varphi = atan2(y, x);
        double strength = sin(M_PI / 2 - angle[1]) * sin(vartheta) * cos(angle[0] - varphi)
                        + cos(M_PI / 2 - angle[1]) * cos(vartheta);
        return alpha + (1.0 - alpha) * strength;
    }
    return 1.0;
}

// Is source position constant at time t
bool SignalGenerator::IsSrcPosConst(const vector<vector<double>>& ss, size_t t, size_t offset) {
    if (t <= offset) return false;
    return ss[t - offset][0] == ss[t - offset - 1][0] &&
           ss[t - offset][1] == ss[t - offset - 1][1] &&
           ss[t - offset][2] == ss[t - offset - 1][2];
}

// Is receiver position constant
bool SignalGenerator::IsRcvPosConst(const vector<vector<vector<double>>>& rr, size_t mic_idx, size_t t) {
    if (t == 0) return false;
    return rr[t][0][mic_idx] == rr[t - 1][0][mic_idx] &&
           rr[t][1][mic_idx] == rr[t - 1][1][mic_idx] &&
           rr[t][2][mic_idx] == rr[t - 1][2][mic_idx];
}

void SignalGenerator::copy_previous_rir(vector<double>& imp, int row_idx, int nsamples) {
    if (row_idx == 0) {
        for (int i = 0; i < nsamples; ++i)
            imp[i * nsamples] = imp[i * nsamples + (nsamples - 1)];
    } else {
        for (int i = 0; i < nsamples; ++i)
            imp[i * nsamples + row_idx] = imp[i * nsamples + (row_idx - 1)];
    }
}


void SignalGenerator::hpf_imp(vector<double>& imp, int row_idx, int nsamples, const HPF& hpf) {
    double Y[3] = {0, 0, 0};
    for (int i = 0; i < nsamples; ++i) {
        double X0 = imp[row_idx + nsamples * i];
        Y[2] = Y[1];
        Y[1] = Y[0];
        Y[0] = hpf.B1 * Y[1] + hpf.B2 * Y[2] + X0;
        imp[row_idx + nsamples * i] = Y[0] + hpf.A1 * Y[1] + hpf.R1 * Y[2];
    }
}


SignalGenerator::Result SignalGenerator::generate(
    const vector<double>& input_signal,
    double c,
    double fs,
    const vector<vector<vector<double>>>& r_path,  // [T][3][M]
    const vector<vector<double>>& s_path,          // [T][3]
    const vector<double>& L,
    const vector<double>& beta_or_tr,
    int nsamples,
    const string& mtype_str,
    int order,
    int dim,
    const vector<vector<double>>& orientation,
    bool hp_filter
) {
    // --- Validate input dimensions ---
    size_t signal_length = input_signal.size();
    if (signal_length != r_path.size() || signal_length != s_path.size())
        throw invalid_argument("Signal length must match time dimension of r_path and s_path.");

    int no_mics = static_cast<int>(r_path[0][0].size());
    if (L.size() != 3)
        throw invalid_argument("Room dimensions (L) must be a 3-element vector.");

    if (!(beta_or_tr.size() == 6 || beta_or_tr.size() == 1))
        throw invalid_argument("Beta must be size 6 or a scalar reverberation time.");

    // --- Compute beta or beta_hat ---
    vector<double> beta(6, 0.0);
    double beta_hat = 0.0;
    double TR = 0.0;

    if (beta_or_tr.size() == 1) {
        TR = beta_or_tr[0];
        double V = L[0] * L[1] * L[2];
        double S = 2.0 * (L[0]*L[2] + L[1]*L[2] + L[0]*L[1]);
        double alpha = 24.0 * V * log(10.0) / (c * S * TR);
        if (alpha > 1.0)
            throw runtime_error("Invalid TR and room dimensions: computed absorption exceeds physical bounds.");
        beta_hat = sqrt(1.0 - alpha);
        fill(beta.begin(), beta.end(), beta_hat);
    } else {
        beta = beta_or_tr;
    }

    // --- Set nsamples if not provided ---
    if (nsamples <= 0) {
        double V = L[0] * L[1] * L[2];
        double S = 2.0 * (L[0]*L[2] + L[1]*L[2] + L[0]*L[1]);
        double alpha = ((1 - pow(beta[0], 2)) + (1 - pow(beta[1], 2))) * L[0] * L[2] +
                       ((1 - pow(beta[2], 2)) + (1 - pow(beta[3], 2))) * L[1] * L[2] +
                       ((1 - pow(beta[4], 2)) + (1 - pow(beta[5], 2))) * L[0] * L[1];
        TR = 24.0 * log(10.0) * V / (c * alpha);
        TR = max(TR, 0.128);  // enforce minimum RT
        nsamples = static_cast<int>(TR * fs);
    }

    // --- Microphone type parsing ---
    vector<char> mtypes(no_mics, 'o');
    if (!mtype_str.empty()) {
        if (mtype_str.size() == 1) {
            fill(mtypes.begin(), mtypes.end(), mtype_str[0]);
        } else if (mtype_str.size() == no_mics) {
            for (int i = 0; i < no_mics; ++i)
                mtypes[i] = mtype_str[i];
        } else {
            throw invalid_argument("mtype string length must be 1 or equal to number of microphones.");
        }
    }

    // --- Orientation parsing ---
    vector<vector<double>> angles(no_mics, vector<double>(2, 0.0));  // [M][2]
    if (!orientation.empty()) {
        if (orientation.size() == 1 && orientation[0].size() == 1) {
            for (int i = 0; i < no_mics; ++i)
                angles[i][0] = orientation[0][0];
        } else if (orientation.size() == 1 && orientation[0].size() == 2) {
            for (int i = 0; i < no_mics; ++i) {
                angles[i][0] = orientation[0][0];
                angles[i][1] = orientation[0][1];
            }
        } else if (orientation.size() == no_mics && orientation[0].size() == 1) {
            for (int i = 0; i < no_mics; ++i)
                angles[i][0] = orientation[i][0];
        } else if (orientation.size() == no_mics && orientation[0].size() == 2) {
            angles = orientation;
        } else {
            throw invalid_argument("Invalid orientation shape.");
        }
    }

    // --- Room dimensions scaling ---
    double cTs = c / fs;
    vector<double> L_scaled(3);
    for (int i = 0; i < 3; ++i)
        L_scaled[i] = L[i] / cTs;

    vector<int> n(3);
    for (int i = 0; i < 3; ++i)
        n[i] = static_cast<int>(ceil(nsamples / (2 * L_scaled[i])));

    const int Tw = 2 * static_cast<int>(round(0.004 * fs));
    vector<double> hanning_window(Tw + 1);
    for (int i = 0; i <= Tw; ++i)
        hanning_window[i] = 0.5 * (1.0 + cos(2.0 * M_PI * (i + Tw / 2.0) / Tw));

    // --- Output array [M][T] ---
    vector<vector<double>> output(no_mics, vector<double>(signal_length, 0.0));

    // High-pass filter config
    HPF hpf;
    hpf.W = 2.0 * M_PI * 100.0 / fs;
    hpf.R1 = exp(-hpf.W);
    hpf.B1 = 2.0 * hpf.R1 * cos(hpf.W);
    hpf.B2 = -pow(hpf.R1, 2);
    hpf.A1 = -(1.0 + hpf.R1);

        // Continue inside generate() method...

    vector<double> LPI(Tw + 1);
    vector<double> imp(nsamples * nsamples, 0.0); // vector<vector<double>> imp(nsamples, vector<double>(nsamples, 0.0));

    vector<double> r(3), s(3);
    vector<double> hu(6), refl(3);

    for (int mic_idx = 0; mic_idx < no_mics; ++mic_idx) {
        const auto& angle = angles[mic_idx];

        // Reset impulse matrix
        std::fill(imp.begin(), imp.end(), 0.0);

        for (size_t t = 0; t < signal_length; ++t) {
            int row_idx_1 = t % nsamples;

            // Normalize receiver position
            for (int i = 0; i < 3; ++i)
                r[i] = r_path[t][i][mic_idx] / cTs;

            // Invariance checks
            bool bSrcInvariant_1 = (t > 0) && IsSrcPosConst(s_path, t, 0);
            bool bRcvInvariant_1 = (t > 0) && IsRcvPosConst(r_path, mic_idx, t);

            int no_rows_to_update = 0;
            if (!(bSrcInvariant_1 && bRcvInvariant_1)) {
                no_rows_to_update = (bRcvInvariant_1 || t == 0) ? 1 : min<int>(t, nsamples);

                for (int row_counter = 0; row_counter < no_rows_to_update; ++row_counter) {
                    int row_idx_2 = mod(row_idx_1 - row_counter, nsamples);
                    bool bSrcInvariant_2 = (row_counter > 0) && IsSrcPosConst(s_path, t, row_counter);

                    if (!bSrcInvariant_2) {
                        for (int i = 0; i < 3; ++i)
                            s[i] = s_path[t - row_counter][i] / cTs;

                        // Clear RIR row
                        for (int i = 0; i < nsamples; ++i)
                            imp[row_idx_2 + nsamples * i] = 0.0;

                        for (int mx = -n[0]; mx <= n[0]; ++mx) {
                            hu[0] = 2 * mx * L_scaled[0];
                            for (int my = -n[1]; my <= n[1]; ++my) {
                                hu[1] = 2 * my * L_scaled[1];
                                for (int mz = -n[2]; mz <= n[2]; ++mz) {
                                    hu[2] = 2 * mz * L_scaled[2];
                                    for (int q = 0; q <= 1; ++q) {
                                        hu[3] = (1 - 2 * q) * s[0] - r[0] + hu[0];
                                        refl[0] = pow(beta[0], abs(mx - q)) * pow(beta[1], abs(mx));
                                        for (int j = 0; j <= 1; ++j) {
                                            hu[4] = (1 - 2 * j) * s[1] - r[1] + hu[1];
                                            refl[1] = pow(beta[2], abs(my - j)) * pow(beta[3], abs(my));
                                            for (int k = 0; k <= 1; ++k) {
                                                hu[5] = (1 - 2 * k) * s[2] - r[2] + hu[2];
                                                refl[2] = pow(beta[4], abs(mz - k)) * pow(beta[5], abs(mz));

                                                if (abs(2 * mx - q) + abs(2 * my - j) + abs(2 * mz - k) <= order || order == -1) {
                                                    double dist = sqrt(pow(hu[3], 2) + pow(hu[4], 2) + pow(hu[5], 2));
                                                    int fdist = static_cast<int>(floor(dist));
                                                    if (fdist < nsamples) {
                                                        for (int idx = 0; idx <= Tw; ++idx)
                                                            LPI[idx] = hanning_window[idx] * sinc(M_PI * (idx - (dist - fdist) - Tw / 2));

                                                        for (int idx = 0; idx <= Tw; ++idx) {
                                                            int pos = fdist - Tw / 2 + idx;
                                                            if (pos >= 0 && pos < nsamples) {
                                                                double strength = sim_microphone(hu[3], hu[4], hu[5], angle, mtypes[mic_idx])
                                                                  * refl[0] * refl[1] * refl[2] / (4 * M_PI * dist * cTs);
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

                        // Apply HPF if enabled
                        if (hp_filter)
                            hpf_imp(imp, row_idx_2, nsamples, hpf);
                    } else {
                        copy_previous_rir(imp, row_idx_2, nsamples);
                    }
                }
            } else {
                copy_previous_rir(imp, row_idx_1, nsamples);
            }

            // Final convolution
            for (int k = 0; k < nsamples; ++k) {
                if (t >= k) {
                    int tmp_imp_idx = mod(row_idx_1 - k, nsamples);
                    output[mic_idx][t] += imp[tmp_imp_idx + nsamples * k] * input_signal[t - k];
                }
            }
        }
    }

    return Result{output, beta_hat};
}

