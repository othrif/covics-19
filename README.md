## Problem
SARS-CoV-2 is spreading rapidly through communities, and healthcare systems are struggling to keep up. Though most 
people infected experience mild disease, some, especially older people and those with underlying health conditions, 
require hospitalization. Many of those infected with SARS-CoV-2 will require hospitalization, and the mortality rate 
is significant and still uncertain. However, hospitals are struggling to cope with the influx of patients and are 
reaching a point of saturation. Without medical assistance and cooperation between countries, far more COVID-19 
patients die. 

## Suggested Solution
To help tackle this problem, we propose a scheme that allows regions to predict how many COVID-19 patients they will 
have to provide in-patient medical care for in the next 3 weeks -- number of weeks still needs to be determined. 
This will be achieved by modelling the infection rate. Regions that will not have enough resources to cope with their 
COVID-19 patients in any given week will be matched with regions that will have a surplus of medical resources in the 
same given time, so that the burden can be shared and more patients can receive life-saving treatment. As the infection 
moves globally, countries that bring their cases under control will be able to provide aid to countries where cases are 
rising rapidly.

## Implementation
### Core functionalities
Our Web App:
   * Predicts the number of COVID-19 cases in a given region that will require hospitalization over the next 3 weeks. 
     Infection rate growth curve over [Hopkins data](https://covid19api.com/#details) will be used for the task.
   * Compares required medical resources to the actual healthcare capacity in the region to determine whether the health 
     system will be overwhelmed or not. Data used comes from 
     [World Bank_1](https://data.worldbank.org/indicator/sh.med.beds.zs), 
     [World Bank_1](https://data.worldbank.org/indicator/SH.MED.NUMW.P3?locations=AU),
     [World Bank_1](https://data.worldbank.org/indicator/SH.MED.PHYS.ZS)
   * Finds the nearest region with excess capacity in the coming weeks, so that medical supplies can be redistributed
   * Provides the above information to the users in an intuitive manner

### Nice to have Functionalities
* Using flights data to better understand the spreading of the virus
    * [Flightradar](https://www.flightradar24.com/data) could be used as data resource
* Extend granularity to city level by using GPS data matching person's location with their health status
   * Person's would need to self assess their health status and communicate it to our App

## Considerations
   * Is it better to move medical resources, or patients themselves?
   * How long does it take to mobilize resources?
   * Which are the political implications? Are countries keen to help each other?
   * Tracking capacity or resources as a whole, or identifying the different available kinds (e.g. masks, ventilation 
     systems, gloves, etc.)
   * Identifying capacity for different kinds of resources would allow countries to have a better picture of what they 
     need and to whom ask for help

## What's next for covics-19
If successful, our project could be moved a step further and be integrated in already existing dashboards used by 
governments' healthcare systems.

## Setting up this Project

### Python Model

#### Create Conda environment
In order to use Python environment used for this project in your local machine, please set it up in this way:

Install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) following the
instructions of the link. Install Anaconda for Python 3.7 version.

Create a conda environment called `covics-19` with following command:
```
conda create -n covics-19 python=3.7 pip
```

To activate this environment, use:
```
conda activate covics-19
```

To deactivate an active environment, use:
```
conda activate covics-19
```

In order to install Python dependencies in this environment, first activate the environment, then install 
`requirements.txt` using pip:
```
conda activate covics-19
pip install -r requirements.txt
```

### Web App

### MongoDB database
Covics-19 database available in [MongoAtlas](https://account.mongodb.com/account/login).

Database name is `covics-19` and data from [Hopkins data](https://covid19api.com/#details) are stored in the collection
called `hopkins` in the format originally provided.

## Modus Operandis
Different tools have used for this project:
* Our [Trello Kanban board](https://trello.com/b/fQuiHhOc/coronalogistic) to track tasks progress
* Our [technical design description](https://docs.google.com/document/d/1FqhjEneccl5-99yW-r4b4qy9Sq-RtJK6f92knWpxSAE/edit)
* Our [Slack](https://slack.com/) workspace `covics-19_team` which we integrated with our GitHub repos