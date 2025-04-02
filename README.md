# ComputerVisionProjectGroup12
Project: Disaster-Resilient Military Base Damage Assessment with Semantic-Segmentation

## Overview

This project utilizes computer vision techniques, specifically semantic segmentation, to assess structural damage in military bases following disasters. By analyzing aerial imagery captured by Unmanned Aerial Vehicles (UAVs), the system aims to provide detailed, pixel-level damage maps, facilitating rapid recovery and strategic planning.

The project focuses on identifying and mapping various types of damage, including cracks, infrastructure failures, and operational disruptions. This detailed analysis allows for a more accurate and efficient assessment of the situation, enabling quicker and more effective response efforts.

## Dataset

The dataset used in this project consists of aerial imagery, similar to that collected after events like Hurricane Michael. It features high-resolution images captured by UAVs, providing a comprehensive overview of military base infrastructure.

**Dataset Characteristics:**

* **Aerial Imagery:** Images captured from UAVs, offering wide-area coverage.
* **Damage Assessment Focus:** The dataset is annotated to facilitate semantic segmentation of structural damage, including:
    * Cracks in buildings and infrastructure.
    * Infrastructure failures (e.g., collapsed structures, damaged roads).
    * Operational disruptions (e.g., blocked access points, damaged equipment).
* **Semantic Segmentation:** The dataset supports pixel-level classification, enabling the creation of detailed damage maps.
* **Similarities to RescueNet:** The dataset shares characteristics with the RescueNet dataset, which includes semantic segmentation labels for:
    * Background
    * Water
    * Building No Damage
    * Building Minor Damage
    * Building Major Damage
    * Building Total Destruction
    * Road-Clear
    * Road-Blocked
    * Vehicle
    * Tree
    * Pool

**Dataset Structure:**

* The dataset is divided into training, validation, and test sets.
* Training set: Approximately 80% of the data.
* Validation and test sets: Approximately 10% of the data each.

## Project Goals

* Develop a robust computer vision system using semantic segmentation for accurate damage assessment.
* Generate detailed damage maps from UAV imagery.
* Provide timely and accurate information to support rapid disaster response and recovery.
* Enhance situational awareness for military personnel in post-disaster scenarios.

* 
## Paper Link

The paper describing the dataset and its collection can be downloaded from this [link](https://www.nature.com/articles/s41597-023-02799-4).










