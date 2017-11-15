#!/bin/bash

rm -r rosserial_libs/*
rosrun rosserial_windows make_libraries.py rosserial_libs
