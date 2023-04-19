# data

## Residential Charging Data
Residential electric vehicle charging datasets from apartment buildings\
Source: https://www.sciencedirect.com/science/article/pii/S2352340921003899

### Data Engineering
Total energyCharged to SOC delta

Assumptions:
1. Filter out charging sessions with less than 1 hour of plug-in time
2. Assume all vehicles are the SAME make and model, meaning the same max battery capacity (80.86 kWh)
3. Assume all vehicles reached 100% endSoc by the end of their charging sessions, which means we could deduct the startSoc
4. Assume all valid sessions need to have a plugin duration that is less than 24hrs, which filtered out aournd 7% of the total session data

Notes
1. Assuming a maximum charging current of 40A from the EVSE and maximum charging power of 9.6kW (240V), all sessions with above assumptions are able to achieve their target energyCharged pr `El_kWh` in their plugin duration, if we sustain the maximum power. 
2. We index the `Start_plugin` and `End_plugout` datetime to 10-minute index out of 1440 minutes in a day
3. 5,810 valid charging sessions remain after the filtering steps