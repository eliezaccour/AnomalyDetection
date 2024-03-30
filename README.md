# AnomalyDetection
A model for anomaly detection on data collected from many elements. An element can respresent anything such as a device, a piece of hardware, software, or other. Each element records a set of measurements, which can be some software metrics, or sensor readings, or any other data points that were generated by the element.

We'll consider a sample relatively complex data set consisting of 3 data files, with samples given below.

**Dataset 1:** Consists of a unique ID identifying the element, a unique ID identifying the measurement (e.g. sensor unique ID), the recorded value, the unit of measurement, and the location of the element (e.g. in which city).

| ElementID | MeasurementID | Value | Unit | Location |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0    | F1    | 0.82    | RH | NYC |
| 0    | B1    | 0.76    | RH | NYC |
| 0    | B2    | 0.68    | RH | NYC |
| 0    | L1    | 0.63    | RH | NYC |
| 0    | L2    | 0.64    | RH | NYC |
| 0    | R1    | 0.87    | RH | NYC |
| 0    | R2    | 0.76    | RH | NYC |
| 1    | F1    | 0.48    | RH | Paris |
| ... |

**Dataset 2:** Consists of several configuration data for the element, along with their values, and IDs.
| ElementID | Configuration | Value | ConfigurationID |
|-----------------|-----------------|-----------------|-----------------|
| 0    | Data Transmission Protocol    | MQTT [LoRaWAN]    | 20
| 0    | Power Source    | Battery [Backup]   | 21 |
| 0    | Control Interface | Wi-Fi [LoRaWAN]    | 24 |
| 1    | Data Transmission Protocol    | HTTP [MQTT]   | 20
| 1    | Power Source    | Solar [Battery]    | 21 |
| 1    | Control Interface | 5G [Wi-Fi]    | 24 |
| ... |

**DataSet 3:** Consists of the element ID again, a serial number unique to the element, and its model number.
| ElementID | SerialNumber | ModelNo |
|-----------------|-----------------|-----------------|
| 0    | SN2024-01-A45B    | IoT-XR100    |
| 1    | SN2024-02-C78D    | IoT-ZY200    |
| ... |
