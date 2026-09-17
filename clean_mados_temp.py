import os
import shutil
import tempfile

def clean_mados_temp():
    temp_dir = tempfile.gettempdir()
    deleted = 0
    for d in os.listdir(temp_dir):
        path = os.path.join(temp_dir, d)
        # Hapus folder mados_job_*
        if d.startswith('mados_job_') and os.path.isdir(path):
            try:
                shutil.rmtree(path)
                print(f"Hapus folder: {path}")
                deleted += 1
            except Exception as e:
                print(f"Gagal hapus {path}: {e}")
        # Hapus file raster besar (opsional)
        elif d.lower().endswith(('.tif', '.tiff', '.zip')) and os.path.isfile(path):
            try:
                os.remove(path)
                print(f"Hapus file: {path}")
                deleted += 1
            except Exception as e:
                print(f"Gagal hapus {path}: {e}")
    if deleted == 0:
        print("Tidak ada file/folder temp MADOS yang ditemukan.")
    else:
        print(f"Selesai. {deleted} file/folder berhasil dihapus.")

if __name__ == "__main__":
    clean_mados_temp()