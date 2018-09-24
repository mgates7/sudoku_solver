/* 
 * File:   main.cpp
 * Author: max
 *
 * Created on September 19, 2018, 9:56 PM
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "SudokuSolver.h"

using namespace std;

typedef vector<int> vec;
typedef vector< vector<int> > d_vec;

void copyV(d_vec v1, d_vec& v2);
void printV(d_vec v, bool pass);
void eraseVal(vec& v, int n);

int main() {
    
    // Initialize array (vector)
    d_vec puzzle(9, vector<int> (9, 0)); // Initialize with 0's
    d_vec temp(9, vector<int> (9, 0)); // Initialize a temp for testing purposes
    bool pass = 0;
    int testNum = 0;

    // Read from text file
    ifstream infile("test_puzzles/evil1.txt");
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            infile >> puzzle[i][j];
        }
    }

    while (~pass) {
        switch (testNum) {
            case 0: // Simple test
                copyV(puzzle, temp);
                cout << "\nRunning simple test..." << endl;
                simpleTest(temp);
                pass = solve(temp);
                printV(temp, pass);
                if (pass)
                    return 0;
            case 1:
                copyV(puzzle, temp);
                cout << "\nRunning test1..." << endl;
                test1(temp);
                pass = solve(temp);
                printV(temp, pass);
                if (pass)
                    return 0;
            case 2:
                copyV(puzzle, temp);
                cout << "\nRunning test2..." << endl;
                test2(temp);
                pass = solve(temp);
                printV(temp, pass);
                if (pass)
                    return 0;
            case 3:
                copyV(puzzle, temp);
                cout << "\nRunning test3..." << endl;
                test3(temp);
                pass = solve(temp);
                printV(temp, pass);
                if (pass)
                    return 0;
            default:
                cerr << "\nPuzzle cannot be solved." << endl;
                return 0;
        }
        testNum++;
    }
    return 0;
}

// Takes v1 and copies contents to v2

void copyV(d_vec v1, d_vec& v2) {
    if (v1.size() != v2.size()) { // error
        cout << "Vectors not the same size. Cannot write." << endl;
        return;
    }
    for (int i = 0; i < v1.size(); i++) {
        for (int j = 0; j < v1.size(); j++) {
            v2[i][j] = v1[i][j];
        }
    }
}

void printV(d_vec v, bool pass) {
    if (pass) {
        cout << "Test PASS\n" << endl;
        cout << "Solution: " << endl;
    } else {
        cout << "Test FAIL\n" << endl;
        cout << "Output: " << endl;
    }

    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v.size(); j++) {
            cout << v[i][j] << " ";
        }
        cout << endl;
    }
}

void eraseVal(vec& v, int n) {
    v.erase(std::remove(v.begin(), v.end(), n), v.end());
}