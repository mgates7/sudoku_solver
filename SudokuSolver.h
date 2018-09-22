/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SudokuSolver.h
 * Author: max
 *
 * Created on September 22, 2018, 4:05 PM
 */

#ifndef SUDOKUSOLVER_H
#define SUDOKUSOLVER_H

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

typedef std::vector<int> vec;
typedef std::vector< std::vector<int> > d_vec;

// Solver functions
bool solve(d_vec v);
void simpleTest(d_vec& v);
void test1(d_vec& v);
void test2(d_vec& v);
void test3(d_vec& v);

#endif /* SUDOKUSOLVER_H */

