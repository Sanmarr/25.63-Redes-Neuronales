from pathlib import Path
import hashlib
import shutil

# ======================================
# CONFIG
# ======================================

ROOT = Path("./skin-dataset-classification/data/Split_smol")

TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "val"

BACKUP_DIR = ROOT / "_duplicates_removed"

EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".webp", ".bmp", ".tif", ".tiff"
}

# ======================================
# HASH
# ======================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)

    return h.hexdigest()

# ======================================
# BUSCAR IMÁGENES
# ======================================

train_images = []
val_images = []

for file in TRAIN_DIR.rglob("*"):
    if file.is_file() and file.suffix.lower() in EXTENSIONS:
        train_images.append(file)

for file in VAL_DIR.rglob("*"):
    if file.is_file() and file.suffix.lower() in EXTENSIONS:
        val_images.append(file)

print(f"\nTrain images: {len(train_images)}")
print(f"Val images:   {len(val_images)}")

# ======================================
# HASHES DE VALIDACIÓN
# ======================================

print("\nCalculando hashes de VAL...")

val_hashes = set()

for idx, img_path in enumerate(val_images, 1):

    if idx % 500 == 0:
        print(f"VAL procesadas: {idx}/{len(val_images)}")

    file_hash = sha256_file(img_path)
    val_hashes.add(file_hash)

# ======================================
# BUSCAR DUPLICADOS EN TRAIN
# ======================================

print("\nBuscando duplicados en TRAIN...")

duplicates = []

for idx, img_path in enumerate(train_images, 1):

    if idx % 500 == 0:
        print(f"TRAIN procesadas: {idx}/{len(train_images)}")

    file_hash = sha256_file(img_path)

    # si existe en val -> eliminar de train
    if file_hash in val_hashes:
        duplicates.append(img_path)

print(f"\nDuplicados encontrados en TRAIN: {len(duplicates)}")

# ======================================
# MOVER DUPLICADOS
# ======================================

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

for dup in duplicates:

    relative = dup.relative_to(ROOT)
    backup_path = BACKUP_DIR / relative

    backup_path.parent.mkdir(parents=True, exist_ok=True)

    # evitar overwrite
    counter = 1
    original_backup = backup_path

    while backup_path.exists():
        backup_path = original_backup.with_stem(
            f"{original_backup.stem}_{counter}"
        )
        counter += 1

    shutil.move(str(dup), str(backup_path))

    print(f"\nMovido:")
    print(f"  {dup}")
    print(f"  -> {backup_path}")

print("\n=====================================")
print("LISTO")
print("Se conservaron las imágenes de VAL")
print("Se movieron los duplicados de TRAIN")
print(f"Backup: {BACKUP_DIR}")
print("=====================================")