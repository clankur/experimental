# %%
from plot_helper import *

# %%
url = "https://app.clear.ml/projects/ce508798b5384eeea472bf6b5357929b/compare-experiments;ids=483668aecfb34f0e98ab13a69270c31f,d4122017e5814f0d94102b8db258b149,8607ffb881a9418cb8276e820e7cf8c5,cb20112968a44b9387f9a5e404f22942,0d26acce03fc46659b7e94ef0f2d4931,c049a27419d84cadb3b6fade2d057051/details"
experiment_ids = get_experiment_ids_from_url(url)
# %%
experiment_ids
# %%
