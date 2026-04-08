from pathlib import Path

root = Path(__file__).resolve().parent
for split in ['dataset/train', 'dataset/validate']:
    labels_dir = root / split / 'labels'
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing directory: {labels_dir}")

    changed = 0
    for label_path in labels_dir.glob('*.txt'):
        text = label_path.read_text(encoding='utf-8').strip()
        new_label = '1' if text else '0'
        if label_path.read_text(encoding='utf-8').strip() != new_label:
            label_path.write_text(new_label + '\n', encoding='utf-8')
            changed += 1

    print(f"Converted {changed} files in {split}")
