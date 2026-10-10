---
title: Contactless Physiological Monitoring using rPPG
emoji: ❤️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
fullWidth: true
pinned: false
short_description: Contactless rPPG estimation of HR, RR, SBP and DBP
---

# Contactless Physiological Monitoring using rPPG

This Space demonstrates a research pipeline for estimating physiological parameters from facial video using remote photoplethysmography (rPPG).

## Outputs

- Heart Rate — primary endpoint
- Respiratory Rate — secondary endpoint
- Systolic Blood Pressure — exploratory endpoint
- Diastolic Blood Pressure — exploratory endpoint

The pipeline uses facial RGB traces extracted from a full-face skin region while excluding the eye regions. Signals are resampled to 30 Hz and analyzed using 30-second windows.

## Research Use Only

This application is a research prototype.

It is **not a medical device** and should not be used for diagnosis, treatment decisions, emergency decisions, or clinical monitoring.