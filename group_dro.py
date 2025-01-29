from b2t_debias.gdro import group_dro, arguments

if __name__ == '__main__':
    print("Training GDRO...")

    args = arguments.get_arguments()
    print(args)
    
    group_dro.main(args)