
def decode_move(s):
    return ord(s[0]) - ord('a'), int(s[1:]) - 1

with open('renjunet_v10_20250716.rif', 'r') as f, open('positions.txt', 'w') as w:
    for line in f.readlines():
        line = line.rstrip()
        if line.startswith('<move>'):
            moves = line[len('<move>'):-len('</move>')].split()
            moves = [decode_move(s) for s in moves]
            w.write(
                    ';'.join(
                        [','.join([str(m[0]), str(m[1])]) for m in moves]
                    )
                    + '\n')

