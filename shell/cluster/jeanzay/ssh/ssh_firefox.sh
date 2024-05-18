#!/bin/bash
#
# Allows a dynamic ssh port forwarding to access the jupyterhub server
# from a remote machine like Grappe.
# It allows to access jupyterhub of jeanzay.
# Needs to be run on from PC.
# Copyright 2023 rho

ssh -N -D 9080 grappe -v

# Other way to create a tunnel
#   ssh -L 7007:jupyterhub.idris.fr:443 grappe
