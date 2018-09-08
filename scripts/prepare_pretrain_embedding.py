import subprocess
import os


if __name__ == '__main__':
    if not os.path.exists('../src/PreTrain/community_detection'):
        os.makedirs('../src/PreTrain/community_detection')
    subprocess.call('./magic -i ../../data/community_detection/com-amazon_agm.txt  -o community_detection/com-amazon_ -nt 20 -c 100 -mi 200', shell=True, cwd='../src/PreTrain/')
    subprocess.call('python format_transform.py ../src/PreTrain/community_detection/com-amazon_final.f.txt ../pre_train/community_detection/com-amazon_pre_train.emb', shell=True)
    subprocess.call('./magic -i ../../data/community_detection/com-dblp_agm.txt  -o community_detection/com-dblp_ -nt 20 -c 100 -mi 200', shell=True, cwd='../src/PreTrain/')
    subprocess.call('python format_transform.py ../src/PreTrain/community_detection/com-dblp_final.f.txt ../pre_train/community_detection/com-dblp_pre_train.emb', shell=True)
    subprocess.call('./magic -i ../../data/community_detection/com-youtube_agm.txt  -o community_detection/com-youtube_ -nt 20 -c 100 -mi 200', shell=True, cwd='../src/PreTrain/')
    subprocess.call('python format_transform.py ../src/PreTrain/community_detection/com-youtube_final.f.txt ../pre_train/community_detection/com-youtube_pre_train.emb', shell=True)
