#!/bin/bash

# {1} -> Host ID returned from starting the cxl-tun binary

sudo ip addr add 10.0.0.${1}/24 dev cxl
sudo sysctl -w net.ipv6.conf.cxl.disable_ipv6=1
sudo ip link set cxl mtu 65472
sudo ip link set cxl up
