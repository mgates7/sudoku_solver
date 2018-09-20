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

using namespace std;

typedef vector<int> vec;
typedef vector< vector<int> > d_vec;

void copyV(d_vec v1, d_vec& v2); // v2 becomes copy of v1
void printV(d_vec v, bool pass);
void eraseVal(vec& v, int n);
void simpleTest(d_vec& v);
bool test(d_vec v);

int main() {

    // Initialize array (vector)
    d_vec puzzle(9, vector<int> (9, 0)); // Initialize with 0's
    d_vec temp(9, vector<int> (9, 0)); // Initialize a temp for testing purposes
    //d_vec solution(9, vector<int> (9, 0)); // Our solution matrix

    ifstream infile("easy.txt");
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            infile >> puzzle[i][j];
        }
    }

    copyV(puzzle, temp);
    simpleTest(temp);
    bool test0 = test(temp);
    printV(temp, test0);
    
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

void simpleTest(d_vec& v1) {
    cout << "Running simple test..." << endl;

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

// Tests for solution

bool test(d_vec v) {
    int sum = 0;

    // Test each row
    for (int i = 0; i < v.size(); i++) {
        sum = 0;
        for (int j = 0; j < v.size(); j++) {
            sum += v[i][j];
        }
        if (sum != 45)
            return 0;
    }

    // Test each column
    for (int i = 0; i < v.size(); i++) {
        sum = 0;
        for (int j = 0; j < v.size(); j++) {
            sum += v[j][i];
        }
        if (sum != 45)
            return 0;
    }

    // Test each sub-matrix
    for (int subX = 0; subX < 2; subX++) {
        for (int subY = 0; subY < 2; subY++) {
            sum = 0;
            for (int ii = subX * 3; ii < subX * 3 + 3; ii++) {
                for (int jj = subY * 3; jj < subY * 3 + 3; jj++) {
                    sum += v[ii][jj];
                }
            }
            if (sum != 45)
                return 0;
        }
    }
    return 1;
}