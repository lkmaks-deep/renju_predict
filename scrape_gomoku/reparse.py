def decode_move(s):
    return ord(s[0]) - ord("a"), int(s[1:]) - 1

with open('results/gomoku_db.txt', 'r') as f:
    with open('results/gomoku_positions.txt', 'w') as w:
        for line in f.readlines():
            moves = [decode_move(s) for s in line.split()]
            w.write(";".join([",".join([str(m[0]), str(m[1])]) for m in moves]) + "\n")