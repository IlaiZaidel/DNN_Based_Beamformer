import argparse
import matplotlib.pyplot as plt
from cues import calc_pdf_cues, plot_pdf3
import numpy as np
import pandas as pd

def main(audio_file1,audio_file2,audio_file3,output_file_name='compare.png'):
    """
    Run the ILD-ITD PDF computation and plot results for a WAV file.
    """
    print(f"[INFO] Processing file: {audio_file1}")
    result = calc_pdf_cues(audio_file1)

    if result is None:
        print(f"[ERROR] Processing failed for file: {audio_file1}")
        return

    wild_itd_pdf1, wild_pdf1, witd_pdf1, ildaxis1, itdaxis1 = result

    print(f"[INFO] Processing file: {audio_file2}")
    result = calc_pdf_cues(audio_file2)

    if result is None:
        print(f"[ERROR] Processing failed for file: {audio_file2}")
        return

    wild_itd_pdf2, wild_pdf2, witd_pdf2, _, _ = result

    print(f"[INFO] Processing file: {audio_file3}")
    result = calc_pdf_cues(audio_file3)

    if result is None:
        print(f"[ERROR] Processing failed for file: {audio_file3}")
        return

    wild_itd_pdf3, wild_pdf3, witd_pdf3, _, _ = result
    # Plot results
    plt.figure(figsize=(8, 6))
    plot_pdf3(
        ild_itd_pdf1=wild_itd_pdf1,
        ild_pdf1=wild_pdf1,
        itd_pdf1=witd_pdf1,
        ild_itd_pdf2=wild_itd_pdf2,
        ild_pdf2=wild_pdf2,
        itd_pdf2=witd_pdf2,
        ild_itd_pdf3=wild_itd_pdf3,
        ild_pdf3=wild_pdf3,
        itd_pdf3=witd_pdf3,
        axis_ild=ildaxis1,
        axis_itd=itdaxis1,
        ild_ff=[], itd_ff=[],color='blue'
    )

    plt.savefig(output_file_name)
    plt.close()

def compre_tau(audio_file_gt,audio_file_blcmv,audio_file_bitse_hrtf,audio_file_bde,lookup_path ='/home/workspace/yoavellinson/extraction_master/RoomImpulseResponseImageSourceExample/python_version/lookuptable_1.txt'):
    lookup = pd.read_csv(lookup_path,delim_whitespace=True)
    result_gt = calc_pdf_cues(audio_file_gt)
    wild_itd_pdf_gt, wild_pdf_gt, witd_pdf_gt, ildaxis, itdaxis = result_gt

    result_blcmv = calc_pdf_cues(audio_file_blcmv)
    wild_itd_pdf_blcmv, wild_pdf_blcmv, witd_pdf_blcmv, _, _ = result_blcmv

    result_bitse_hrtf = calc_pdf_cues(audio_file_bitse_hrtf)
    wild_itd_pdf_bitse_hrtf, wild_pdf_bitse_hrtf, witd_pdf_bitse_hrtf, _, _ = result_bitse_hrtf

    result_bde = calc_pdf_cues(audio_file_bde)
    wild_itd_pdf_bde, wild_pdf_bde, witd_pdf_bde, _, _ = result_bde

    itd_peak_gt = np.argmax(witd_pdf_gt)

    itd_peak_blcmv = np.argmax(witd_pdf_blcmv)

    itd_peak_bitse_hrtf = np.argmax(witd_pdf_bitse_hrtf)

    itd_peak_bde = np.argmax(witd_pdf_bde)

    delta_itd_blcmv = abs(itdaxis[itd_peak_blcmv] - itdaxis[itd_peak_gt])

    delta_itd_bitse_hrtf = abs(itdaxis[itd_peak_bitse_hrtf] - itdaxis[itd_peak_gt])

    delta_itd_bde = abs(itdaxis[itd_peak_bde] - itdaxis[itd_peak_gt])

    subset_blcmv = lookup[lookup['axis_idx']==itd_peak_blcmv]
    min_key_blcmv = subset_blcmv["az"].min()
    max_key_blcmv = subset_blcmv["az"].max()
    if np.isnan(min_key_blcmv):
        if itd_peak_blcmv > 52:
            theta_hat_blcmv='>90'
        else:
            theta_hat_blcmv='<-90'
    else:
        theta_hat_blcmv = f'[{min_key_blcmv},{max_key_blcmv}]'

    subset_bitse_hrtf = lookup[lookup['axis_idx']==itd_peak_bitse_hrtf]
    min_key_bitse_hrtf = subset_bitse_hrtf["az"].min()
    max_key_bitse_hrtf = subset_bitse_hrtf["az"].max()
    if np.isnan(min_key_bitse_hrtf):
        if itd_peak_bitse_hrtf > 52:
            theta_hat_bitse_hrtf='>90'
        else:
            theta_hat_bitse_hrtf='<-90'
    else:
        theta_hat_bitse_hrtf = f'[{min_key_bitse_hrtf},{max_key_bitse_hrtf}]'
    theta_hat_bitse_hrtf = f'[{min_key_bitse_hrtf},{max_key_bitse_hrtf}]'


    subset_bde = lookup[lookup['axis_idx']==itd_peak_bde]
    min_key_bde = subset_bde["az"].min()
    max_key_bde = subset_bde["az"].max()
    if np.isnan(min_key_bde):
        if itd_peak_bde > 52:
            theta_hat_bde='>90'
        else:
            theta_hat_bde='<-90'
    else:
        theta_hat_bde = f'[{min_key_bde},{max_key_bde}]'
    theta_hat_bde = f'[{min_key_bde},{max_key_bde}]'


    df = pd.DataFrame({
        "Method": ["BLCMV Beamformer", "BDE-BiTSE", "Bi-TSE-HRTF"],
        "delta_ITD": [f"{abs(delta_itd_blcmv)}",f"{abs(delta_itd_bitse_hrtf)}",f"{abs(delta_itd_bde)}"],
        "theta_hat[°]": [f"{theta_hat_blcmv}",f"{theta_hat_bitse_hrtf}",f"{theta_hat_bde}"],
    })    
    return print(df)

def theta_from_tau(tau, delta_tau, c=343.0, d=0.18):
    x = (tau + delta_tau) * c / d  # supposed to be cosθ
    # stable acos via atan2; also return a saturation flag
    sat = np.abs(x) > 1.0
    theta = np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - x*x)), x)
    return np.rad2deg(theta), sat

if __name__ == "__main__":

    # audio_file_bitsehrtf = '/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/mixs_ys_rev_two_speakers_15/y_hat_1_9_az_300_elev_-10_sisdr_-5.099.wav'
    # audio_file_bde = '/home/workspace/yoavellinson/Direction-based-BiTSE/exp/save_model_yoav_rev_two_speakers/output_audio_enhance/y_hat_1_9_thetha_60.wav'
    # name = 'y1_9_compare.png'
    # main(audio_file_bitsehrtf,audio_file_bde,name)

    # audio_file_bitsehrtf = '/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/mixs_ys_rev_two_speakers_15/y_hat_2_9_az_270_elev_-10_sisdr_-0.688.wav'
    # audio_file_bde = '/home/workspace/yoavellinson/Direction-based-BiTSE/exp/save_model_yoav_rev_two_speakers/output_audio_enhance/y_hat_2_9_thetha_90.wav'
    # name = 'y2_9_compare.png'
    # main(audio_file_bitsehrtf,audio_file_bde,name)
    audio_file_gt = '/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/mixs_ys_rev_two_speakers_15/y1_9_az_300_elev_-10.wav'
    audio_file_blcmv = '/dsi/scratch/users/yoavellinson/outputs/blcmv_rev_wsj0/y_hat_2_9_az_300_elev_-9_sisdr_4.987060822972586.wav'
    audio_file_bitsehrtf = '/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/mixs_ys_rev_two_speakers_15/y_hat_1_9_az_300_elev_-10_sisdr_-5.099.wav'
    audio_file_bde = '/home/workspace/yoavellinson/Direction-based-BiTSE/exp/save_model_yoav_rev_two_speakers/output_audio_enhance/y_hat_1_9_thetha_60.wav'
    name = 'y1_9_compare_3_min.png'
    # compre_tau(audio_file_gt,audio_file_blcmv,audio_file_bitsehrtf,audio_file_bde)
    main(audio_file_blcmv,audio_file_bitsehrtf,audio_file_bde,name)

    audio_file_blcmv='/dsi/scratch/users/yoavellinson/outputs/blcmv_rev_wsj0/y_hat_1_9_az_270_elev_-4_sisdr_-5.129888323787484.wav'
    audio_file_bitsehrtf = '/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/mixs_ys_rev_two_speakers_15/y_hat_2_9_az_270_elev_-10_sisdr_-0.688.wav'
    audio_file_bde = '/home/workspace/yoavellinson/Direction-based-BiTSE/exp/save_model_yoav_rev_two_speakers/output_audio_enhance/y_hat_2_9_thetha_90.wav'
    name = 'y2_9_compare_3_min.png'
    main(audio_file_blcmv,audio_file_bitsehrtf,audio_file_bde,name)


#y2 /dsi/scratch/users/yoavellinson/outputs/blcmv_rev_wsj0/y_hat_1_9_az_270_elev_-4_sisdr_-5.129888323787484.wav
#y1 /dsi/scratch/users/yoavellinson/outputs/blcmv_rev_wsj0/y_hat_2_9_az_300_elev_-9_sisdr_4.987060822972586.wav