import os
import sys
import traceback
import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import cpu_count

try:
    from fairseq import checkpoint_utils
except ImportError:
    print("Библиотека fairseq не найдена. Пожалуйста, установите ее перед запуском скрипта.")
    sys.exit(1)

# Example: 
#       python3 train_index_print.py mi-test v2

exp_dir_arg = sys.argv[1] if len(sys.argv) > 1 else "mi-test"
version_arg = sys.argv[2] if len(sys.argv) > 1 else "v2"

now_dir = os.getcwd()
n_cpu = cpu_count()

def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    
    if not os.path.exists(feature_dir) or len(os.listdir(feature_dir)) == 0:
        return "Директория с особенностями не найдена или пуста. Пожалуйста, выполните извлечение особенностей перед обучением индекса."
    
    listdir_res = list(os.listdir(feature_dir))
    
    try:
        npys = []
        for name in sorted(listdir_res):
            phone = np.load("%s/%s" % (feature_dir, name))
            npys.append(phone)
        big_npy = np.concatenate(npys, 0)
    except Exception as e:
        error_message = f"Ошибка при загрузке и объединении файлов .npy: {str(e)}"
        print(error_message)
        return error_message
    
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    
    if big_npy.shape[0] > 1e4:
        print("Trying doing kmeans %s shape to 1k centers." % big_npy.shape[0])
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=1000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            print(info)
    
    min_size = 100  # Минимальный размер big_npy для создания индекса
    if big_npy.shape[0] < min_size:
        return f"Недостаточно данных для создания индекса. Требуется как минимум {min_size} векторов, найдено {big_npy.shape[0]}."
    
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(4 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    print("%s,%s" % (big_npy.shape, n_ivf))
    
    try:
        index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
        print("training")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)
        
        index_file = "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19)
        faiss.write_index(index, index_file)
        print("adding")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])
        
        index_file = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19)
        faiss.write_index(index, index_file)
        print(f"Индекс успешно создан и сохранен в файл: {index_file}")
        return "OK"
    except Exception as e:
        error_message = f"Ошибка при создании или сохранении индекса Faiss: {str(e)}"
        print(error_message)
        return error_message

print(train_index(exp_dir_arg, version_arg))
