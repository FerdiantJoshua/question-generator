with open('Datasets/SQuAD/v2.0/dev-v2.0.json', 'r') as f_in:
    buffer = f_in.read(1024000)
    print(buffer)
    with open('Datasets/SQuAD/v2.0/dev-v2.0-test.txt', 'w') as f_out:
        f_out.write(buffer)
