from hq_seg.trainer import run_train

if __name__ == '__main__':
    import sys
    run_train(sys.argv[1], 1000, device='cuda:0')
    pass