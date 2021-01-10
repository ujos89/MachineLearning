# Solar power generation prediction
## Step0. Environment setting
- Ubuntu 20.04
- Python 3.6

## Step1. Data preprocessing


## Step2. Predict 5 elements
**Elements to be predicted**
> **DHI(W/m<sup>2</sup>)**  : Diffuse Horizontal Irradiance<br>
> **DNI(W/m<sup>2</sup>)**  : Direct Normal Irradiance<br>
> **WS(m/s)** : Wind Speed<br>
> **RH(%)** : Relative Humidity<br>
> **T(<sup>o</sup>C)**  : Temperature<br>

- BPTT(Backpropagation Through Time) => WS, RH, T
- Sunrise Time, Sunset Time & BPTT => DHI, DNI

## Step3. Predict target
**Target to be predicted**
> **Generation(kW)**  : Solar power generation<br>

**Supervised Learning**
> **Neural Network**<br>
> **KNN(K Nearest Neighbor)**<br>

## Step4. Analysis
- 
