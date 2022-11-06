import numpy as np
import os
import pandas as pd
import psutil

from datetime import datetime
from pandarallel import pandarallel

from freeholdforecast.tasks.pandarallel_sample_static import func_apply as func

# pandarallel.initialize()

cpu_count = psutil.cpu_count(logical=True)
is_local = os.getenv("APP_ENV") == "local"
pandarallel.initialize(
    nb_workers=cpu_count,
    progress_bar=is_local,
    use_memory_fs=False,
    verbose=1,
)

print("Creating dataframe")
df_size = int(5e5)
df = pd.DataFrame(dict(a=np.random.randint(1, 8, df_size), b=np.random.rand(df_size)))


print("Running apply")
start = datetime.now()
res = df.apply(func, axis=1)
print(f"Run time: {(datetime.now()-start).total_seconds():.2f} seconds")

print("Running parallel apply")
start = datetime.now()
res_parallel = df.parallel_apply(func, axis=1)
print(f"Run time: {(datetime.now()-start).total_seconds():.2f} seconds")

print(f"Equals: {res.equals(res_parallel)}")
