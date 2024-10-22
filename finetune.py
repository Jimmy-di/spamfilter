def main():
    args = create_parser()

def create_parser():
    parser = argparse.ArgumentParser(description='Calculating activation statistics')
    
    parser.add_argument('--dataset', default='dataset/', type=str, help='The file from which the prompts are loaded to calculate the statistics (default: \'prompts/additional_laion_prompts.csv\').')
    parser.add_argument('--model_name', default='gpt2-large', type=str, help='Name of the model used for finetuning')
    parser.add_argument('--model_path', default='model/checkpoints/model.ptr', type=str, help='Name of the model used for finetuning')
    parser.add_argument(
        '--load_model',
        default=False,
        action='store_true',
        help='Load a model from model_path.'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()