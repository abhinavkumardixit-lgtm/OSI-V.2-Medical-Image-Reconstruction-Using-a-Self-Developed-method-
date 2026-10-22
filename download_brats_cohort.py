"""
TME-FUSE BraTS 2023 Cohort Downloader CLI.
Downloads and pre-processes genuine clinical mpMRI scans from the
MICCAI BraTS 2023 Glioma Challenge into normalized 64^3 patches.
"""

from tme_fuse.download_real_data import main

if __name__ == "__main__":
    main()
