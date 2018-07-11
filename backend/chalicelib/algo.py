import random
import pandas as pd

# new_data and existing_samples should be of type pd.DataFrame
def reservior_sampling(sample_size, new_data,
                       existing_record_count=0, existing_samples=pd.DataFrame()):
    new_data = new_data.reset_index(drop=True)
    samples = existing_samples.reset_index(drop=True)
    replaced_indexes = {}
    for index, record in new_data.iterrows():
        if samples.shape[0] < sample_size:
            samples = samples.append(record, ignore_index=True)
            replaced_indexes[index] = index
        else:
            r = random.randint(0, existing_record_count + index)
            if r < sample_size:
                samples.loc[r] = record
                replaced_indexes[r] = index
    samples.reset_index(inplace=True)
    remaining_indexes = set(range(len(new_data))) - set(replaced_indexes.values())
    return (samples, remaining_indexes)
