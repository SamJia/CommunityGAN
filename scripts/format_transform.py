import sys
import os


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python bigclam_out2cdgan_in.py bigclam_out_filename output_filename')
    bigclam_out_filename = sys.argv[1]
    output_filename = sys.argv[2]
    out_dirname = os.path.dirname(output_filename)
    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname)
    with open(bigclam_out_filename) as fp, open(output_filename, 'w') as out_fp:
        _, C = fp.readline().split()
        C = int(C)
        out_lines = []
        for line in fp:
            line = line.strip().split('\t')
            if line[0][0] == 'w':
                continue
            node_id = line[0].strip('d')
            line = line[1]
            embedding = [0] * C
            if line != '[]':
                line = line[2:-2].split(')(')
                for pair in line:
                    pair = pair.split(',')
                    embedding[int(pair[0])] = float(pair[1])
            out_lines.append('%s %s\n' % (node_id, ' '.join(str(i) for i in embedding)))
        out_fp.write('%d %d\n' % (len(out_lines), C))
        out_fp.writelines(out_lines)
