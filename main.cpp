#include <iostream>
#include <fstream>
#include <algorithm> // remove and remove_if
#include <vector> // the general-purpose vector container

using namespace std;

void print_matrix(int matrix[9][9]);
bool simple_test(int matrix[9][9]);

int main() {

    int matrix[9][9];
    // Input sudoku puzzle here...blanks will be represented by some symbol other than 1-9
    ifstream infile("example.txt");
    for(int i=0; i<9; i++)
        for (int j=0; j<9; j++)
            infile >> matrix[i][j];
	simple_test(matrix);
    return 0;

}

void print_matrix(int matrix[9][9]) {
    for(int i=0; i<9; i++) {
        for(int j=0; j<9; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

bool simple_test (int matrix[9][9]) {
    cout << "Input: " << endl;
    print_matrix(matrix);

    cout << "\nRunning simple_test..." << endl;
    int sub_x;
    int sub_y;
    bool zero_flag = 0;
    int zero_count = 0;

    do {
        zero_flag=0; // reset flag
        for(int i=0; i<9; i++) {
            for(int j=0; j<9; j++) {
                vector <int> v = {1,2,3,4,5,6,7,8,9}; // for determing solution for matrix point
                sub_x = j/3;
                sub_y = i/3;
                if(matrix[i][j] == 0) {
                    zero_flag = 1; // set flag
                    for(int m=0; m<9; m++) {
                        if(matrix[i][m] != 0) {
                            v.erase(remove( v.begin(), v.end(), matrix[i][m] ), v.end() );
                        }
                        if (v.size() == 1) {
                            matrix[i][j] = v[0];
                            goto end;
                        }
                    }
                    for(int n=0; n<9; n++) {
                        if(matrix[n][j] != 0) {
                            v.erase(remove(v.begin(), v.end(), matrix[n][j]), v.end());
                        }
                        if(v.size() == 1) {
                            matrix[i][j] = v[0];
                            goto end;
                        }
                    }
                    for (int y=sub_y*3; y<(sub_y*3)+3; y++) {
                        for (int x=sub_x*3; x<(sub_x*3)+3; x++) {
                            if(matrix[y][x] != 0) {
                                v.erase(remove(v.begin(), v.end(), matrix[y][x]), v.end());
                            }
                            if(v.size() == 1) {
                                matrix[i][j] = v[0];
                                goto end;
                            }
                        }
                    }
                    zero_count++;
                    if(zero_count == 81){
                        return 0;
                        cout << "Simple test FAIL\n" << endl;
                        cout << "Output: " << endl;
                        print_matrix(matrix);
                    }
                }
end:
                zero_count = 0;
                ;
            }
        }
    } while(zero_flag);
    cout << "Simple test PASS\n" << endl;
    cout << "Solution: " << endl;
    print_matrix(matrix);
    return 1;
}
