import os

def update_class_in_annotations(dataset_path):
    # Проходим по всем папкам train, valid, test
    for subset in ['train', 'valid', 'test']:
        subset_path = os.path.join(dataset_path, subset, 'labels')
        if os.path.exists(subset_path):
            for filename in os.listdir(subset_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(subset_path, filename)
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                    with open(file_path, 'w') as file:
                        for line in lines:
                            parts = line.strip().split()
                            if parts[0] == '0':
                                parts[0] = '6'
                            file.write(' '.join(parts) + '\n')
                    print(f"Updated class in {file_path}")

if __name__ == '__main__':
    dataset_path = 'D:\\123'
    update_class_in_annotations(dataset_path)
    print("Class update completed.")