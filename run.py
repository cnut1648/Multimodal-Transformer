from itertools import zip_longest
import dotenv
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

OmegaConf.register_new_resolver(
    "ratio", lambda a, b: a / (a+b)
)
blocks1, blocks2 = map(list, [range(6), range(10)])
exchange_block_ids = [0, -3]
value1, value2 = 0, 0
ready_to_exchange1, ready_to_exchange2 = False, False
i1, i2 = 0, 0
# loop until both get to the last block
while i1 < len(blocks1) or i2 < len(blocks2):
    for exchange_block_id in exchange_block_ids:
        if i1 < len(blocks1) and blocks1[i1] is blocks1[exchange_block_id]:
            ready_to_exchange1 = True
        if i2 < len(blocks2) and blocks2[i2] is blocks2[exchange_block_id]:
            ready_to_exchange2 = True
    if ready_to_exchange1 and ready_to_exchange2:
        print(f"exchange {value1} & {value2}")
        ready_to_exchange1 = False
        ready_to_exchange2 = False
    
    # if ready to exchange but not exchange, wait until the other is ready
    # can't go above the last one
    if i1 < len(blocks1) and not ready_to_exchange1:
        value1 = i1
        i1 += 1
    if i2 < len(blocks2) and not ready_to_exchange2:
        value2 = i2 + 100
        i2 += 1

print(value1, value2)
assert value1 == blocks1[-1]
assert value2 == blocks2[-1] + 100


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # return

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src import utils

    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    config = utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()