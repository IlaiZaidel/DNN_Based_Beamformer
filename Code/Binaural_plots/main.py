import argparse
import matplotlib.pyplot as plt
from cues import calc_pdf_cues, plot_pdf
import matplotlib.image as mpimg
from pathlib import Path

def main(audio_file):
    """
    Run the ILD-ITD PDF computation and plot results for a WAV file.
    """
    print(f"[INFO] Processing file: {audio_file}")
    result = calc_pdf_cues(audio_file)

    if result is None:
        print(f"[ERROR] Processing failed for file: {audio_file}")
        return

    wild_itd_pdf, wild_pdf, witd_pdf, ildaxis, itdaxis = result

    # Plot results
    plt.figure(figsize=(8, 6))
    plot_pdf(
        ild_itd_pdf=wild_itd_pdf,
        ild_pdf=wild_pdf,
        itd_pdf=witd_pdf,
        axis_ild=ildaxis,
        axis_itd=itdaxis,
        ild_ff=[], itd_ff=[]
    )
    plt.savefig('/Users/yoavellinson/Downloads/python_version/mix_0_new.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a WAV file and plot ILD-ITD PDFs.")
    args = parser.parse_args()
    args.audio_file='/Users/yoavellinson/Downloads/python_version/mix_0.wav'
    main(args.audio_file)

    # dir = Path('/home/workspace/yoavellinson/extraction_master/complex_nn/extraction/mixs_ys_rev_two_speakers_15')
    # files = dir.glob('*.wav')
    # for file in files:
    #     args.audio_file=str(file)
    #     main(args.audio_file)
    # mixs = dir.glob('mix_*.png')
    # for mix in mixs:
    #     idx = mix.name.split('_')[-1].split('.')[0]
    #     mix_path=mix
    #     s1_path = list(mix.parent.glob(f'y_hat_1_{idx}_*.png'))[0]
    #     s2_path = list(mix.parent.glob(f'y_hat_2_{idx}_*.png'))[0]
    #     mix_img = mpimg.imread(mix_path)
    #     s1_img = mpimg.imread(s1_path)
    #     s2_img = mpimg.imread(s2_path)

    #     # Create the figure with 3 subplots in one row
    #     fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    #     # Plot each image
    #     for ax, img, title in zip(axes, [mix_img, s1_img, s2_img], ["Mix", "S1", "S2"]):
    #         ax.imshow(img)
    #         ax.set_title(title)
    #         ax.axis("off")

    #     # Adjust layout and save
    #     plt.tight_layout()
    #     plt.savefig(dir/f"plots/combined_{idx}.png", dpi=300)  # Saves the combined image
    #     # plt.show()


    # dir = Path('/home/workspace/yoavellinson/Direction-based-BiTSE/exp/save_model_yoav_rev_two_speakers')
    # files = dir.glob('**/*.wav')
    # # for file in files:
    # #     args.audio_file=str(file)
    # #     main(args.audio_file)
    # mixs = dir.glob('**/mix_*.png')
    # for mix in mixs:
    #     idx = mix.name.split('_')[-1].split('.')[0]
    #     mix_path=mix
    #     s1_path = list((mix.parent.parent/'output_audio_enhance/').glob(f'y_hat_1_{idx}_*.png'))[0]
    #     s2_path = list((mix.parent.parent/'output_audio_enhance/').glob(f'y_hat_2_{idx}_*.png'))[0]
    #     mix_img = mpimg.imread(mix_path)
    #     s1_img = mpimg.imread(s1_path)
    #     s2_img = mpimg.imread(s2_path)

    #     # Create the figure with 3 subplots in one row
    #     fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    #     # Plot each image
    #     for ax, img, title in zip(axes, [mix_img, s1_img, s2_img], ["Mix", "S1", "S2"]):
    #         ax.imshow(img)
    #         ax.set_title(title)
    #         ax.axis("off")

    #     # Adjust layout and save
    #     plt.tight_layout()
    #     plt.savefig(dir/f"plots/combined_{idx}.png", dpi=300)  # Saves the combined image