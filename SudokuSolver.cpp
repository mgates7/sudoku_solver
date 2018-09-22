/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SudokuSolver.cpp
 * Author: max
 * 
 * Created on September 22, 2018, 4:05 PM
 */

#include "SudokuSolver.h"

using namespace std;

void copyV(d_vec v1, d_vec& v2);
void printV(d_vec v, bool pass);
void eraseVal(vec& v, int n);

void simpleTest(d_vec& v1) {

    vec testV(9);
    int subX;
    int subY;
    int count1 = 0;
    int count2 = 0;
    bool flag = 0;

    while (flag == 0) {
        count1 = 0;
        for (int i = 0; i < v1.size(); i++) {
            for (int j = 0; j < v1.size(); j++) {
                subX = i / 3;
                subY = j / 3;
                //cout << subX << " " << subY << endl;
                if (v1[i][j] == 0) {
                    count1++;
                    //cout << count1 << " " << count2 << endl;
                    testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};

                    // Test vertical
                    for (int ii = 0; ii < v1.size(); ii++)
                        if (v1[ii][j] != 0)
                            eraseVal(testV, v1[ii][j]);
                    // Test horizontal
                    for (int jj = 0; jj < v1.size(); jj++)
                        if (v1[i][jj] != 0)
                            eraseVal(testV, v1[i][jj]);
                    // Test sub-matrix
                    for (int ii = subX * 3; ii < subX * 3 + 3; ii++)
                        for (int jj = subY * 3; jj < subY * 3 + 3; jj++)
                            if (v1[ii][jj] != 0)
                                eraseVal(testV, v1[ii][jj]);
                    if (testV.size() == 1) {
                        v1[i][j] = testV[0];
                        goto end;
                    }
                }
end:
                ;
            }
        }
        //cout << count1 << " " << count2 << endl;
        if (count1 == count2)
            flag = 1;
        count2 = count1;
    }
}

// Brute force with trying all 9 combs for each 0

void test1(d_vec& v) {

    vec testV(9);
    testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    d_vec tempDV = v; // make copy
    bool pass = 0;

    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v.size(); j++) {
            if (v[i][j] == 0) {
                v = tempDV;
                for (int ii = 0; ii < 9; ii++) {
                    v = tempDV;
                    v[i][j] = testV[ii];
                    simpleTest(v);
                    pass = solve(v);
                    if (pass)
                        return;
                }
            }
        }
    }
}

// More advanced brute force

void test2(d_vec& v) {
    vec testV(9);
    testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    d_vec tempDV = v; // make copy
    bool pass = 0;
    vector<int*> zeroPoints; // location of zeros

    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v.size(); j++)
            if (v[i][j] == 0)
                zeroPoints.push_back(&v[i][j]);

    for (int i = 0; i < zeroPoints.size(); i++) {
        for (int j = i + 1; j < zeroPoints.size(); j++) {
            for (int ii = 0; ii < 9; ii++) {
                for (int jj = 0; jj < 9; jj++) {
                    *zeroPoints[i] = testV[ii];
                    *zeroPoints[j] = testV[jj];
                    copyV(v, tempDV);
                    simpleTest(tempDV);
                    pass = solve(tempDV);
                    if (pass) {
                        copyV(tempDV, v);
                        return;
                    }
                }
            }
            // reset back to zero
            *zeroPoints[i] = 0;
            *zeroPoints[j] = 0;
        }
    }
}


// Even more advanced brute force

void test3(d_vec& v) {
    vec testV(9);
    testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    d_vec tempDV = v; // make copy
    bool pass = 0;
    vector<int*> zeroPoints; // location of zeros

    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v.size(); j++)
            if (v[i][j] == 0)
                zeroPoints.push_back(&v[i][j]);

    for (int i = 0; i < zeroPoints.size(); i++) {
        for (int j = i + 1; j < zeroPoints.size(); j++) {
            for (int k = j + 1; k < zeroPoints.size(); k++) {
                //cout << i << " " << j << " " << k << endl;
                for (int ii = 0; ii < 9; ii++) {
                    for (int jj = 0; jj < 9; jj++) {
                        for (int kk = 0; kk < 9; kk++) {
                            
                            *zeroPoints[i] = testV[ii];
                            *zeroPoints[j] = testV[jj];
                            *zeroPoints[k] = testV[kk];
                            copyV(v, tempDV);
                            simpleTest(tempDV);
                            pass = solve(tempDV);
                            if (pass) {
                                copyV(tempDV, v);
                                return;
                            }
                        }
                    }
                }
                // reset back to zero
                *zeroPoints[i] = 0;
                *zeroPoints[j] = 0;
                *zeroPoints[k] = 0;
            }
        }
    }
}

// Tests for solution

bool solve(d_vec v) {
    vec testV(9);
    testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Test each row
    for (int i = 0; i < v.size(); i++) {
        testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (int j = 0; j < v.size(); j++) {
            if (v[i][j] == 0)
                return 0;
            eraseVal(testV, v[i][j]);
        }
        if (testV.size() != 0)
            return 0;
    }

    // Test each column
    testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < v.size(); i++) {
        testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (int j = 0; j < v.size(); j++) {
            if (v[j][i] == 0)
                return 0;
            eraseVal(testV, v[j][i]);
        }
        if (testV.size() != 0)
            return 0;
    }

    // Test each sub-matrix
    testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int subX = 0; subX < 2; subX++) {
        for (int subY = 0; subY < 2; subY++) {
            testV = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            for (int ii = subX * 3; ii < subX * 3 + 3; ii++) {
                for (int jj = subY * 3; jj < subY * 3 + 3; jj++) {
                    eraseVal(testV, v[ii][jj]);
                }
            }

            if (testV.size() != 0)
                return 0;
        }
    }
    return 1;
}


