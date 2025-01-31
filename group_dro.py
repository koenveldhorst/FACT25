from b2t_debias.gdro import group_dro, arguments
from codecarbon import EmissionsTracker

if __name__ == '__main__':
    print("Training GDRO...")

    args = arguments.get_arguments()
    print(args)

    # run gdro with footprint tracker
    with EmissionsTracker(project_name="B2T", measure_power_secs=10, output_file=f"{args.dataset}.csv") as tracker:
        group_dro.main(args)
