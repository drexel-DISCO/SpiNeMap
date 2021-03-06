/*
 * Noxim - the NoC Simulator
 *
 * (C) 2005-2010 by the University of Catania
 * For the complete list of authors refer to file ../doc/AUTHORS.txt
 * For the license applied to these sources refer to file ../doc/LICENSE.txt
 *
 * This file represents the top-level testbench
 */

#ifndef __NOXIMNOC_H__
#define __NOXIMNOC_H__

#include "GlobalRoutingTable.h"
#include "GlobalTrafficTable.h"
#include "GlobalTrafficTrace.h"
#include "NeuralNetworkParser.h"
#include "Segment.h"
#include "Tile.h"
#include <fstream>
#include <iostream>
#include <string>
#include <systemc.h>
#include <vector>
//#include "GlobalSelectionTable.h"
#include "Channel.h"
#include "Hub.h"
#include "TokenRing.h"

using namespace std;

template <typename T>
struct sc_signal_NSWE {
    sc_signal<T> east;
    sc_signal<T> west;
    sc_signal<T> south;
    sc_signal<T> north;
};

template <typename T>
struct sc_signal_NSWEH {
    sc_signal<T> east;
    sc_signal<T> west;
    sc_signal<T> south;
    sc_signal<T> north;
    sc_signal<T> to_hub;
    sc_signal<T> from_hub;
};

SC_MODULE(NoC) {
    // I/O Ports
    sc_in_clk clock;   // The input clock for the NoC
    sc_in<bool> reset; // The reset signal for the NoC

    // Signals and variables for mesh network
    // Signals
    sc_signal_NSWEH<bool> **req;
    sc_signal_NSWEH<bool> **ack;
    sc_signal_NSWEH<Flit> **flit;
    sc_signal_NSWE<int> **free_slots;

    // NoP
    sc_signal_NSWE<NoP_data> **nop_data;

    // Matrix of tiles
    Tile ***t;

    map<int, Hub *> hub;
    map<int, Channel *> channel;

    TokenRing *token_ring;

    // Global tables
    GlobalRoutingTable grtable;
    GlobalRoutingFile gr_file;
    GlobalTrafficTable gttable;
    GlobalTrafficTrace gtrtable;
    NeuralNetworkParser nn_parser;
    //    GlobalSelectionTable gsltable;

    // Constructor

    SC_CTOR(NoC) {

        buildMesh();
    }

    // Support methods
    Tile *searchNode(const int id) const;

  private:
    void buildMesh();
};

// Hub * dd;

#endif
