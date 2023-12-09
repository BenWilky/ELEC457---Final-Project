**ELEC 457 - Final Project**

**Ben Wilkinson - 72583727**

This python notebook contains all the required information and procedures to complete a thourough analysis of a power system that includes generation, storage and loads.

PyPSA is the backbone of this notebook and it's capabilities are leveraged throughout to ensure "accurate" results. PyPSA provides an overhead structure that allows for designer, engineers and hobbyist to create power systems through simple classes. These classes can be seperated into the categories of loads, generators, storage, and transmission. Extremely detailed information can attached to these classes including nominal, max, minimum output, ramp up time/cost, ramp down time and cost, max charge states, charge rate, temperature limits, and many many more variables. A deep dive into all the available variables can be found in PyPSA's documentation.

For this use case, PyPSA was put to use with Highs Solver, a optimization tool, to create a monte carlo or stochastic optimization if you wish. Both load and wind and solar variability (C.F.) we're adjusted with a uniform distribution to captures a range of cases that could be used to evaluate pricing, availabilioty and capacity planning for the future. To be considered here is that transmission was not taken into account with this model, that is, we assume an infinite bus with zero losses that all generation, storage and laods are connected to.


In the following sections between blocks of code, descriptions of what is being done and why can be found. If relevant, notes on how to edit the source code and why you may want to will also laid out in these text blocks.

To get started though, you must first have some basic libraries and programs installed. First, install Visual Studio Code, the latest version of python and reopen this the entire project folder in VS Code.

The first time you set this program up it will take some time to load all the libraries and you will need to complete the extra steps below:
1. Open a terminal in VS Code (once you have installed it and python)
2. type in the terminal "pip install jupyter notebooks virtualenv" - this will take some time but allow it finish without stopping
3. Next we need to setup and open our "Virtual Environment", inside of this vitual environment all of the required dependicies are stored.
4. To open and initate the virtual environment we must return to to the terminal in VS Code and type " .\venv\Scripts\activate" - This will activate the venv that is required.
5. At the top right of this notebook you will now see a place to select the "interpreter" we want to use. You must click there and click the interepreter venv(Python X.X.X) where X.X is a reference to the version of python you have installed
6. Finally, you can click "Run All" at the top of this window. The first time running this script will take quite alot of time as all the required dependicies are being installed in the background. I would recommend running the script for the first time and leaving it while you have a coffee (Not to worry though, this extremely long wait only occurs on the first run of this script on a new computer)

**Now let's get started with the program and breakdown what each step is doing/what you as the user can change**


**Imports**

Here we are importing all the required libraries that we installed above.

You most likely will never have to change this unless you want to add more functionality through different libraries. I might add though, that the libraries imported here can do many things including plotting, matrix manipulation/math, optimization and more. 

To learn about these libraries, just type the name of any of them into google and look at their documentation. (Probably a years worth of reading if you want to know them in and out)


```python
import pypsa
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
```

**Variables**

Here is where we as the user can change different variables based on the conditions we want.

Don't want nuclear or hydro? Just set them both to **False**. Only want hydro, set only hydro to **True**. 

What to adjust the price of carbon so it is more or less expensive to run gas generators? Adjust the cost of carbon per ton with the variable carbon_cost. If you set the carbon market to **False** this expresses the global constraint that there must be zero cabron emissions. If you are okay with carbon emissions but want it to cost serious dollars to emit then you can adjust the carbon_cost as needed while leaving the **carbon_market = True**.

We also have a load growth factor variable which grows the load by that amount per year to the year of interest. So if the base load data is 2014 and we want to investigate capacity planning in the year 2050 the calculation will be load*((2050-2014)*(1+load_growth)).

We aslo have a conversion rate to convert the technology pricing from Euros to Canadians Dollars. This can change all the time obviously.

Finally, the variables year and month are what allow for the user to select the year we are interested in investigating and a specific month (a specific month is chosen as this allows for easy plotting and viewing but does not play a role in the overall optimization as all months are taken into account when optimizing the system).

PyPSA provides technology data in their Github for years 2025, 2030, 2035, 2040, 2045, and 2050. These are the only years can select and using a differernt year will result in an error as no data is avaialble.

This technology data can be found at: https://github.com/PyPSA/technology-data


```python
#User Variables
#Carbon market - yes or no  (True or False)
carbon_market = True
#Hydro - yes or no (True or False)
hydro = True
#Nuclear - yes or no (True or False)
nuclear = True
#Eur to CAD conversion rate
conversion_rate = 1.44
#Cost of carbon (CAD/ton)
carbon_cost = 500
#Load growth factor per year (%/100)
load_growth = 0.02

#Year of analysis - 2025, 2030, 2035, 2040, 2045, 2050 only
year = 2025
#Month of analysis - 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11 or 12 - Must be in quotes
month = "08"
#Setting up the date for the analysis
date = f"{year}-{month}"
#Collecting technology data for given year from PYPSA technology data repository
url = f"https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_{year}.csv"
#Loading data into a dataframe called costs
costs = pd.read_csv(url, index_col=[0, 1])
```

**Unit Conversion**

Here we are converting some of the units in the data we just downloaded. 
Instead of using euro's we're using CAD instead. We are also converting kW to MW, the base unit for PyPSA.


```python
#Converting imported data to standards required for PyPSA analysis - Need MW and CAD units for each technology
costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
costs.unit = costs.unit.str.replace("/kW", "/MW")
costs.loc[costs.unit.str.contains("EUR/"), "value"] *= conversion_rate
costs.unit = costs.unit.str.replace("EUR/", "CAD/")
```

Here we need to setup some default values. This is because if a technology is missing some parameters ("NaN") and they are passed to the solver the entire solver will fail.

Below we fill all empty cells in the cost dataframe with the defaults shown below to deal with the possibilty for errors later


```python
defaults = {
    "FOM": 0,
    "VOM": 0,
    "efficiency": 1,
    "fuel": 0,
    "investment": 0,
    "lifetime": 25,
    "CO2 intensity": 0,
    "discount rate": 0.15,
}
costs = costs.value.unstack().fillna(defaults)
```

**Pricing adjustments**

Here we can edit the price of some of the technologies to better match what we expect to see in real life.

PyPSA technologies database can be very generous (to say the least) with cost/megawatt of some technologies like hydo and nuclear. To better adjust the model to the acutal cost of both nuclear and hydro we can use some real world examples of projects to tune the cost/MW. 

For example, we know that Site C will produce 1100 MW and the cost has exploded to 16 Billion Dollars (no comment), therefore, the cost is roughly 14.5 Million/MW. We also know that the cost of nuclear from projects across the North America is on average 8.1 Million/MW. 


```python
#Adjust Hydro Pricing based on site C dam (16 Billion CAD$/1100MW) = 14.5 Million/MW
costs.at["hydro", "investment"] = 14.5*1e6 #CAD/MW
#Adjust Nuclear pricing Based on Nuclear Power Plant Average
costs.at["nuclear", "investment"] = 8.1*1e6 #CAD/MW
```

This function called annuity takes both the rate and the lifetime of the project as variables to calculate marginal cost of each technology in the present value 


```python
def annuity(r, n):
    return r / (1.0 - 1.0 / (1.0 + r) ** n)
```

Here is where where we calculate the marginal cost of each technology. We have fixed operating cost (FOM) and variable operating cost (VOM) that can be calculated using the annuity function above and data provided by PyPSA technology resource. 

Here we calcualte the marginal cost as the VOM + the cost of fuel divided by the efficiency. We can then take the annuity+FOM/100 and multiply it by the cost of investment to determine the capaital cost of the project. We then insert this value into the cost dataframe that can be used later by the solver. 

We are also adding to the marginal cost of the CCGT the cost of carbon per ton. This is because if we allow carbon emissions and we say the cost per ton is $x, we need to account for those emissions with the CCGT.


```python
costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]
costs.at["CCGT", "marginal_cost"] = costs.at["CCGT", "marginal_cost"] + carbon_cost
annuity = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
costs["capital_cost"] = (annuity + costs["FOM"] / 100) * costs["investment"]
```

Here we import both the load data for BC from 2014 and the solar and onshore wind variability data for Washignton State (Closest available data source) for 2014. Other variability data is available including California and Canada (averaged). You can chose which source you would like to use by editing the file name in quotes. If you want to use Canada instead of Washignton you need to edit the file name in the quotes from ""timeseries-data-wash.csv" to "timeseries-data-canada.csv".

Here we also change the year from 2014 to the year of interest to create less confusion later.

Variability data from https://model.energy/


```python
#Creating a new dataframe called ts_data and reading the comma seperated values from the time series data file into the dataframe
ts_data = pd.read_csv("timeseries-data-wash.csv",index_col=0,parse_dates=True)
#Replace 2014 year with year of analysis
ts_data.index = ts_data.index.map(lambda t: t.replace(year=year))
#Print the first 5 rows of the dataframe
ts_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>load</th>
      <th>onwind</th>
      <th>solar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2025-01-01 00:00:00</th>
      <td>7067</td>
      <td>0.025</td>
      <td>0.089</td>
    </tr>
    <tr>
      <th>2025-01-01 01:00:00</th>
      <td>6890</td>
      <td>0.022</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2025-01-01 02:00:00</th>
      <td>6732</td>
      <td>0.014</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2025-01-01 03:00:00</th>
      <td>6660</td>
      <td>0.007</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2025-01-01 04:00:00</th>
      <td>6666</td>
      <td>0.004</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>



This is where we now apply the load growth factor to the load data that was just imported above. Depending on the load_growth factor chosen above we can see the growth of the load when comparing the table above and below this block.


```python
#Calculate the factor to adjust the load based on the load growth factor
factor = (year - 2014)*load_growth
#Adjust the load based on the factor
ts_data["load"] += ts_data["load"]*factor
#Print the first 5 rows of the dataframe now that the load has been adjusted
ts_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>load</th>
      <th>onwind</th>
      <th>solar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2025-01-01 00:00:00</th>
      <td>8621.74</td>
      <td>0.025</td>
      <td>0.089</td>
    </tr>
    <tr>
      <th>2025-01-01 01:00:00</th>
      <td>8405.80</td>
      <td>0.022</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2025-01-01 02:00:00</th>
      <td>8213.04</td>
      <td>0.014</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2025-01-01 03:00:00</th>
      <td>8125.20</td>
      <td>0.007</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2025-01-01 04:00:00</th>
      <td>8132.52</td>
      <td>0.004</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have done all the required data importing and manipulation it is now time to setup the PyPSA network and being loading all the technolgies and data into the network for solving.

First, we setup the PyPSA network with psa = pypsa.network and we add a single electrical bus called "electricity"


```python
psa = pypsa.Network()
psa.add("Bus", "electricity")
```

Next we want to setup the time "snapshots" in the PyPSA network. Therefore, we load the ts_data set into psa with psa.set_snapshot. The ts_data dataset is load and variability data that we had looked at above. In this case though, all we care about for the snapshots is the time data it also has which is hourly data for one year.


```python
psa.set_snapshots(ts_data.index)
```

**Technology**

Now we get into what I considered the most important part of this program that can provide us with a huge amount of control over our generation and storage.

In the following blocks we begin to add each "technology" to the PyPSA network. To add a technology we need to create an instance of the technology class with "psa.add()"

We can now provide a few major pieces of information for the technology. For instance in the code block below we are adding a Closed Cycle Gas Generator to the electricity network. First we specify its a generator, second we call it a CCGT this related to the technology data we downloaded above and it must match exactly. Third, we tell it what bus we want it to connect to and we call the carrier a CCGT. Fourth, we provide the class with some data, that is the capital and marginal cost, its efficiency and if we want its nominal power to be extendable (i.e. do you want to allow the optimizer to expand the nominal power generation). 

You can also add other information to this class, one example could be telling the optimizer it must have a minimum of x amount of power availble from some technology or the starup cost is y and the ramp down time is z. There are roughly 30 options you can provide to the optimizer and the oppurtunities are almost endless. 

To find all the possible option a link to the PyPSA documentation that outlines the options is here: https://pypsa.readthedocs.io/en/latest/components.html#generator


```python
psa.add(
    #Type - generator
    "Generator",
    #Name of technology - CCGT
    "CCGT",
    #Bus attachment - electricity
    bus="electricity",
    #Nice Name
    carrier="CCGT",
    #Input data
    capital_cost=costs.at["CCGT", "capital_cost"],
    marginal_cost=costs.at["CCGT", "marginal_cost"],
    efficiency=costs.at["CCGT", "efficiency"],
    p_nom_extendable=True,
)
```

We are continuing to add technologies here. If we allow nuclear to be included as a technology then we follow a similar class setup to above but with nuclear instead of CCGT.


```python
if nuclear is True:
    psa.add(
        "Generator",
        "nuclear",
        bus="electricity",
        carrier="nuclear",
        capital_cost=costs.at["nuclear", "capital_cost"],
        marginal_cost=costs.at["nuclear", "marginal_cost"],
        efficiency=costs.at["nuclear", "efficiency"],
        p_nom_extendable=True,
    )
```

Same thing here with hydro. If hydro==True. Then add hydro to the PyPSA network.


```python
if hydro is True:
    psa.add(
        "Generator",
        "hydro",
        bus="electricity",
        carrier="hydro",
        capital_cost=costs.at["hydro", "capital_cost"],
        marginal_cost=costs.at["hydro", "marginal_cost"],
        efficiency=costs.at["hydro", "efficiency"],
        p_nom_extendable=True,
    )
```

Again, the same as above but now for wind and solar technology. We add each as there own technology.


```python
for tech in ["onwind", "solar"]:
    psa.add(
        "Generator",
        tech,
        bus="electricity",
        carrier=tech,
        p_max_pu=ts_data[tech],
        capital_cost=costs.at[tech, "capital_cost"],
        marginal_cost=costs.at[tech, "marginal_cost"],
        efficiency=costs.at[tech, "efficiency"],
        p_nom_extendable=True,
    )
```

Now here we are adding storage units. A grid with renewables such as wind and solar need to have energy storage avaialble to ensure that power is available when the sun isn't out or the wind isn't blowing. In this case we follow similar steps to the generation technology above but instead we call this a storage unit and specify what type of storage we want to use. 

Here we have decided to use battery storage to backup the grid. We say each storage unit can provide power to the grid for 4 hours and we have a total of 6 of them. We also provide the efficieny of storing and dispatching the energy to the grid and we allow the nominal power to be extendable if needed. Here we also proivde minimum amount of storage that must be created to ensure that there is adequate grid storage available (80% of the average load over the course of the year in interest).


```python
psa.add(
    "StorageUnit",
    "battery storage",
    bus="electricity",
    carrier="battery storage",
    max_hours=4,
    capital_cost=costs.at["battery inverter", "capital_cost"]
    + 6 * costs.at["battery storage", "capital_cost"],
    efficiency_store=costs.at["battery inverter", "efficiency"],
    efficiency_dispatch=costs.at["battery inverter", "efficiency"],
    p_nom_extendable=True,
    p_nom_min=0.8*ts_data["load"].mean(),
    cyclic_state_of_charge=True,
)
```

We can also do the same for Hydrogen Storage if we want as well. Here it has been commented out, so it will not be added to the network, but if we want we could remove the "#" signs and have Hydrogen storage available as well. 

Adding hydrogen storage follows almost exactly the same steps as the battery storage and both could be added to the network if preferred. 


```python
# psa.add(
#     "StorageUnit",
#     "pumped hydro",
#     bus="electricity",
#     carrier="Pumped-Storage-Hydro-store",
#     max_hours=8,
#     capital_cost=costs.at["Pumped-Storage-Hydro-bicharger", "capital_cost"]
#     + 2 * costs.at["Pumped-Storage-Hydro-store", "capital_cost"],
#     efficiency_store=costs.at["Pumped-Storage-Hydro-bicharger", "efficiency"],
#     efficiency_dispatch=costs.at["Pumped-Storage-Hydro-bicharger", "efficiency"],
#     p_nom_extendable=True,
#     p_nom_min=0.8*ts_data["load"].mean(),
#     cyclic_state_of_charge=True,
# )
```

This next block is for the global constraint on carbon emissions. If we say there is no carbon market (i.e. carbon_market = False) then we set up global constraint for the solver telling it there most be zero carbon emissions. This while obviously change the outcome of the optimization as any carbon emitting technology will be ignored.


```python
if carbon_market is False:
    psa.add(
         "GlobalConstraint",
         "CO2Limit",
         carrier_attribute="co2_emissions",
         sense="<=",
         constant=0,
    )
```

Now we are into some data plotting. First let's look at the wind and solar variablity factors for one month. The month and year can be chosen in the user variable section at the top of this document.


```python
#Get the data from the network and put it in a dataframe
df = psa.generators_t.p_max_pu.loc[date] 
#Setup the plotly figure
fig = px.line(df, title=f"Wind and Solar Capacity in {date}").update_layout(yaxis_title="Capacity Factor (pu)", xaxis_title="Date")
#Plot the figure
fig.show()
```



This small funciton calculates the total system cost. It takes into account the capital expenditures and operating expenditures for each technology and provides a sum cost for each technology as a dataframe. This funciton is used later on to allow for plotting of cost


```python
def system_cost(psa):
    tsc = psa.statistics.capex() + psa.statistics.opex(aggregate_time="sum")
    return tsc.droplevel(0).div(1e6)  # million CAD$/a
```

**The optimizer**

Finally, this is the bread and butter of the system. This is where we setup the random uniform distribution for the load  and the variability data that will then be inputted into the optimizer and solved for each case.

With the variable "runs" we can set how many runs we would like to evaluate. The higher the number the longer the optimization takes. 

Next we get two uniform distributions one for the load data that is between 0.5 and 1.5 and one for the variabilty data that is bewteen 0.2 and 1.2.

We then setup a few dataframes that can be used to store the outputs from the optimizer as it running. 

Next we begin the optimization. Firstly, we adjust the variability data by a factor from the distribution then we add the load data to the network that is also adjusted by a factor from the distribution. We then start the optimization with "psa.optimize(solver_name="highs")", telling PyPSA to use the Highs solver. Once this optimization is complete we save all the relevant data in the dataframes loads, gen, out and storage. We then remove the old load from the network and restart the process again with new factors for each run until we have completed all the runs.

Once all the optimization runs are complete all data will be stored in the dataframes discussed above. This data can now be plotted and manipulated to tell us about the power system and how it reacts to different scenarios.


```python
#How many runs dowe want to do
runs = 10
#Unifrom distribution of load and capacity factor
dist = np.random.uniform(0.5,1.5,runs)
dist_cf = np.random.uniform(0.2,1.2,runs)    
#Dataframes for data collection
out = pd.DataFrame()
gen = pd.DataFrame()
loads = pd.DataFrame()
storage = pd.DataFrame()
norm_pu = psa.generators_t.p_max_pu
#Run the model for each load and capacity factor
for i in range(runs):
    psa.generators_t.p_max_pu = dist_cf[i]*psa.generators_t.p_max_pu
    psa.add(
        "Load",
        "demand",
        bus="electricity",
        p_set=ts_data["load"]*dist[i],
    )
    psa.optimize(solver_name="highs")
    loads[i] = ts_data["load"]*dist[i]
    gen[i] = psa.generators.p_nom_opt
    out[i] = system_cost(psa)
    if not psa.storage_units.empty:
        storage[i] = psa.storage_units.p_nom_opt
    psa.remove("Load", "demand")
    
#Get the mean, low and high values from the distribution
mean = dist.mean()
low = dist.min()
high = dist.max()
```

    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 22.99it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 65.72it/s]
    INFO:linopy.io: Writing time: 0.78s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 1.56e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 23.32it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 65.72it/s]
    INFO:linopy.io: Writing time: 0.76s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 1.64e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 22.95it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 59.06it/s]
    INFO:linopy.io: Writing time: 0.79s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 1.21e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 22.69it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 67.05it/s]
    INFO:linopy.io: Writing time: 0.78s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 1.19e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 22.76it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 66.34it/s]
    INFO:linopy.io: Writing time: 0.77s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 1.96e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 23.01it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 66.01it/s]
    INFO:linopy.io: Writing time: 0.77s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 2.05e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 21.73it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 59.07it/s]
    INFO:linopy.io: Writing time: 0.82s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 2.33e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 23.56it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 65.72it/s]
    INFO:linopy.io: Writing time: 0.76s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 2.23e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 21.48it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 63.34it/s]
    INFO:linopy.io: Writing time: 0.83s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 2.17e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 20.84it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 55.94it/s]
    INFO:linopy.io: Writing time: 0.86s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 2.39e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    

Now we can start plotting the data we collected. First lets look at the distribution of load data for one month.


```python
fig = px.line(loads.loc[date]).update_layout(yaxis_title="Load (MW)", xaxis_title="Date")
fig.show()
```



Next lets plot the range of capacities calculated by the optimizer for each technology. In the plots we cna see the high, low, median, upper and lower fences for each technology. 


```python
fig = go.Figure().update_layout(yaxis_title="Capacity (MW)", xaxis_title="Technology")
x1=gen.loc["CCGT"]
fig.add_trace(go.Box(y=x1, name="CCGT"))
if nuclear is True:
    x2=gen.loc["nuclear"]
    fig.add_trace(go.Box(y=x2, name="Nuclear"))
if hydro is True:
    x3=gen.loc["hydro"]
    fig.add_trace(go.Box(y=x3, name="Hydro"))
x4=gen.loc["solar"]
fig.add_trace(go.Box(y=x4, name="Solar"))
x5=gen.loc["onwind"]
fig.add_trace(go.Box(y=x5, name="Wind"))
x6=storage.loc["battery storage"]
fig.add_trace(go.Box(y=x6, name="Battery Storage"))
# x7=storage.loc["pumped hydro"]
# fig.add_trace(go.Box(y=x7, name="Pumped Hydro"))
fig.show()
```



We can also plot the cost distribution for each technology. This is similar to the plot above but instead of capacities, its cost. 


```python
#out.plot(kind="box", figsize=(12, 6), ylabel="System Cost (million CAD$/a)")
fig = go.Figure().update_layout(yaxis_title="System Cost (million CAD$)", xaxis_title="Technology")
x1=out.loc["CCGT"]
fig.add_trace(go.Box(y=x1, name="CCGT"))
if nuclear is True:
    x2=gen.loc["nuclear"]
    fig.add_trace(go.Box(y=x2, name="Nuclear"))
if hydro is True:
    x3=gen.loc["hydro"]
    fig.add_trace(go.Box(y=x3, name="Hydro"))
x4=out.loc["solar"]
fig.add_trace(go.Box(y=x4, name="Solar"))
x5=out.loc["onwind"]
fig.add_trace(go.Box(y=x5, name="Wind"))
x6=out.loc["battery storage"]
fig.add_trace(go.Box(y=x6, name="Battery Storage"))
# x7=out.loc["Pumped-Storage-Hydro-store"]
# fig.add_trace(go.Box(y=x7, name="Pumped Hydro"))
fig.show()
```



To simplify the analysis, we can also plot the total system cost as a distribution. This again is similar to the plots above but is now for the total system cost instead.


```python
fig = px.box(out.sum(), title="System Cost Distribution").update_layout(yaxis_title="System Cost (million CAD$)",xaxis={'visible': False})
fig.show()
```



Here we are plotting the total system cost per optimiztion run. This gives us an idea of how the total cost fluctuates between runs. 


```python
fig = px.line(out.sum(), title="System Cost by Scenario").update_layout(yaxis_title="System Cost (million CAD$)",xaxis_title="Scenario",showlegend=False)
fig.show()
```



We can also look at the distribution of loads for each run with a box plot similar to the ones before. 


```python
fig = px.box(loads.sum().div(1e3), title=f"Load Distribution for {year}").update_layout(yaxis_title="Load (MW)", xaxis={'visible': False})
fig.show()
```



Here we are running the optimizer again but this time for the average load of all the runs. This will give us a good middle ground case.


```python
load=ts_data["load"]*mean
psa.generators_t.p_max_pu = norm_pu
psa.add(
    "Load",
    "demand",
    bus="electricity",
    p_set=load,
)
psa.optimize(solver_name="highs")
```

    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 22.69it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 62.45it/s]
    INFO:linopy.io: Writing time: 0.8s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 1.85e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    




    ('ok', 'optimal')



Now that we have reoptimized the network for the average load case we can extract the data from the optimized network and plot the dispatch curve.

The code block below collects all the data in a dataframe "results_mean" and plots it with the load curve overlaid on top.

This data may vary well now have solar and wind generation technologies now because the variability has now been set back to its original value. This change will most likely cause the data to be adjusted in comparison to what has been seen above.

It may have also decided that now CCGT's are not worth it because we can make up the difference with solar, wind, and battery technologies. This is especially true if the cost per ton of carbon is high ior there is no carbon market.


```python
result_mean = pd.concat([psa.generators_t.p.loc[date].div(1e3), psa.storage_units_t.p.loc[date].div(1e3)], axis=1, join='inner')
fig = px.bar(
    result_mean,
    title=f"Generation and Storage for {date}",
    labels={"value": "Power (GW)"},
).update_layout(yaxis_title="Power (GW)", xaxis_title="Date",bargap=0)
fig.add_trace(go.Scatter(x=result_mean.index,y=load.loc[date].div(1e3), name="Load", mode='lines'))
fig.show()
```



Now we can examine some of the data from the optimization of the average case. First we can look at the nominal optimized power for each technology in GW.


```python
psa.generators.p_nom_opt.div(1e3)  # GW
```




    Generator
    CCGT       2.284328
    nuclear    8.981474
    hydro     -0.000000
    onwind     2.806860
    solar      4.449623
    Name: p_nom_opt, dtype: float64




```python
psa.storage_units.p_nom_opt.div(1e3)  # MW
```




    StorageUnit
    battery storage    6.886548
    Name: p_nom_opt, dtype: float64



We can also look at the TWh output for each technology.


```python
psa.snapshot_weightings.generators @ psa.generators_t.p.div(1e6)  # TWh
```




    Generator
    CCGT        2.631204
    nuclear    69.858936
    hydro       0.000000
    onwind      4.676993
    solar       5.951780
    Name: generators, dtype: float64



The cost of each technology in millions of Canadian Dollars.


```python
(psa.statistics.capex() + psa.statistics.opex(aggregate_time="sum")).div(1e6)
```




                 carrier        
    StorageUnit  battery storage     2174.285148
    Generator    CCGT                1860.894734
                 hydro                  0.000000
                 nuclear            13116.552774
                 onwind               728.815494
                 solar                660.014043
    dtype: float64



Any emmisions we may have if the global constraint on emissions is not true. 


```python
emissions = (
    psa.generators_t.p
    / psa.generators.efficiency
    * psa.generators.carrier.map(psa.carriers.co2_emissions)
)  # t/h
```


```python
psa.snapshot_weightings.generators @ emissions.sum(axis=1).div(1e6)  # Mt
```




    0.0



We can also plot a pie chart that shows us the cost of each technology.


```python
cost = system_cost(psa)
for x in cost.index:
    if cost[x] == 0:
        cost = cost.drop(x)
    else:
        continue
fig = px.pie(cost, values=cost, names=cost.index, title="System Cost Breakdown")
fig.show()
```




```python
demand = psa.snapshot_weightings.generators @ psa.loads_t.p_set.sum(axis=1)
```


```python
system_cost(psa).sum() * 1e6 / demand.sum()
```




    223.44460425701598



We can now repeat a similar process for the low of the load distribution.


```python
psa.remove("Load", "demand")
load=ts_data["load"]*low
psa.generators_t.p_max_pu = norm_pu
psa.add(
    "Load",
    "demand",
    bus="electricity",
    p_set=load,
)
psa.optimize(solver_name="highs")
psa.remove("Load", "demand")
```

    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 22.02it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 59.30it/s]
    INFO:linopy.io: Writing time: 0.81s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 1.18e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    

And plot the dispatch curve for the low of the load distribution.


```python
result_low = pd.concat([psa.generators_t.p.loc[date].div(1e3), psa.storage_units_t.p.loc[date].div(1e3)], axis=1, join='inner')
fig = px.bar(
    result_low,
    title=f"Generation and Storage for {date}",
    labels={"value": "Power (GW)"},
).update_layout(yaxis_title="Power (GW)", xaxis_title="Date",bargap=0)
fig.add_trace(go.Scatter(x=result_low.index,y=load.loc[date].div(1e3), name="Load", mode='lines'))
fig.show()
```



Finally, we can do the same for the high of the load distribution.


```python
load=ts_data["load"]*high
psa.generators_t.p_max_pu = norm_pu
psa.add(
    "Load",
    "demand",
    bus="electricity",
    p_set=load,
)
psa.optimize(solver_name="highs")
psa.remove("Load", "demand")
```

    INFO:linopy.model: Solve problem using Highs solver
    INFO:linopy.io:Writing objective.
    Writing constraints.: 100%|[38;2;128;191;255m██████████[0m| 14/14 [00:00<00:00, 22.00it/s]
    Writing continuous variables.: 100%|[38;2;128;191;255m██████████[0m| 7/7 [00:00<00:00, 63.34it/s]
    INFO:linopy.io: Writing time: 0.81s
    INFO:linopy.solvers:Log file at C:\Users\Ben\AppData\Local\Temp\highs.log.
    INFO:linopy.constants: Optimization successful: 
    Status: ok
    Termination condition: optimal
    Solution: 69894 primals, 157254 duals
    Objective: 2.37e+10
    Solver model: available
    Solver message: optimal
    
    INFO:pypsa.optimization.optimize:The shadow-prices of the constraints Generator-ext-p-lower, Generator-ext-p-upper, StorageUnit-ext-p_dispatch-lower, StorageUnit-ext-p_dispatch-upper, StorageUnit-ext-p_store-lower, StorageUnit-ext-p_store-upper, StorageUnit-ext-state_of_charge-lower, StorageUnit-ext-state_of_charge-upper, StorageUnit-energy_balance were not assigned to the network.
    

And plot the dispatch curve for the high of the distribution.


```python
result_high = pd.concat([psa.generators_t.p.loc[date].div(1e3), psa.storage_units_t.p.loc[date].div(1e3)], axis=1, join='inner')
fig = px.bar(
    result_high,
    title=f"Generation and Storage for {date}",
    labels={"value": "Power (GW)"},
).update_layout(yaxis_title="Power (GW)", xaxis_title="Date",bargap=0)
fig.add_trace(go.Scatter(x=result_high.index,y=load.loc[date].div(1e3), name="Load", mode='lines'))
fig.show()
```



We have now come to the end of this file. We have covered quite alot of scenarios that carry a significant amount of data. Obviously, more can always be done and the options are almost limitless with PyPSA. More can be built on top of this model and understanding it's basics are the key to increasing the accuracy and breadth of this model. The PyPSA domcumentation has a wealth of information available and is a good starting point if you are ever stuck. With that being said I am also available to field question and help with design. Don't hesitate to reach out to me at ben@wilkinsonsix.com or 604-306-8571
