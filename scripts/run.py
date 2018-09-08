import subprocess


if __name__ == '__main__':
    subprocess.call('python community_gan.py dataset com-amazon n_emb 100 motif_size 3', shell=True, cwd='../src/CommunityGAN/')
    subprocess.call('python community_gan.py dataset com-youtube n_emb 100 motif_size 3', shell=True, cwd='../src/CommunityGAN/')
    subprocess.call('python community_gan.py dataset com-dblp n_emb 100 motif_size 3', shell=True, cwd='../src/CommunityGAN/')
