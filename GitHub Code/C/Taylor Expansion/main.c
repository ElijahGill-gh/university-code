/*
 * Assessment 2 for PHY2027
 * Author: Elijah Gill
 * Due Date: 17/11/2023
*/

/*
 * Program description: Creating a function that calculates the Taylor series of sin(x), which calls a separate function
 *                      to handle the factorials. The function is then used to find the Taylor series of an array of inputs
 *                      and compare them to the actual sin(x) values.
 *                      Every integer angle between 0 and 360 degrees is used to calculate sin(x) and the Taylor series, these values
 *                      are then written to a text file that will be used to plot the values in a different program to be compared.
 *                      Each of these tasks is presented as an option for the user to select.
 * 
 *                      I scored 74% on this project.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double const PI = 3.141592654; // Pi appears frequently so a constant is created to easily use it
int N_MAX = 5; // Used to avoid magic numbers in the arrays

// Declaring functions
int factorial(int x);
double sum_sin_series(int x,int n);

int main() {
    // Creating the array and required variables
    char option; // Used in the menu to check the entered character
    int single_angle, num_terms;
    int angle_arr[N_MAX];
    double sin_arr[N_MAX], tay_arr[N_MAX];
    // Making a menu for the user to choose what to do
    printf("=================================================\n");
    printf("============ Taylor series of sin(x) ============\n");
    printf("=================================================\n");
    printf(" --- Please choose an option ---\n");
     // Giving the user 3 different options to choose, or to quit the program instead
    do {
        printf(" - Enter '1' to use the Taylor series function on its own.\n");
        printf(" - Enter '2' to enter 5 angles at a time.\n");
        printf(" - Enter '3' to print sin(x) and its Taylor series between 0 and 360 degrees.\n");
        printf(" - Enter 'q' to quit the program.\n");
        scanf(" %c", &option);
        while (getchar() != '\n');
        if ((option != '1') && (option != '2') && (option != '3') && (option != 'q')) {
            printf("Please choose an available option.\n\n");
        }
    }
    while ((option != '1') && (option != '2') && (option != '3') && (option != 'q'));
    // If option 1 is chosen, the user is asked for an angle and the number of terms to calculate the Taylor series to
    if (option == '1') {
        printf("Please enter an integer angle in degrees.\n");
        scanf("%d", &single_angle);
        printf("Please enter the number of terms you want to calculate the Taylor series to.\n");
        scanf("%d", &num_terms);
        if (num_terms < 18) {
            printf("The Taylor series for sin(x) at %d degrees with %d terms = %g", single_angle, num_terms, sum_sin_series(single_angle, num_terms));
        } else {
            // If the number of terms is higher than what can be used in the function, an error message is displayed
            printf("The maximum number of terms allowed is 17! The Taylor series has not been calculated.\n");
        }
    }
    // If option 2 is chosen, the user is prompted to enter 5 angles and the number of terms to be used in the calculation
    else if (option == '2') {
        printf(" - Please enter 5 integer angles in degrees.\n");
        scanf(" %d %d %d %d %d", &angle_arr[0], &angle_arr[1], &angle_arr[2], &angle_arr[3], &angle_arr[4]);
        printf(" - Please enter the number of terms n, you want the Taylor series to calculate to.\n");
        scanf(" %d", &num_terms);
        printf("\nAngles given: %d, %d, %d, %d, %d [degrees]\n", angle_arr[0], angle_arr[1], angle_arr[2], angle_arr[3], angle_arr[4]);
        printf("Number of terms specified: %d\n", num_terms);
        if (num_terms < 18) {
            // Using the values given with the sum_sin_series() function and the sine function, saving the results to arrays
            for (int i=0; i<N_MAX; i++) {
                tay_arr[i] = sum_sin_series(angle_arr[i], num_terms);
                sin_arr[i] = sin(angle_arr[i]*(PI/180));
            }
            // Printing out the values calculated for the Taylor series
            printf("\nFor the Taylor series to %d terms, the values of sin(x) are:\n", num_terms);
            for (int i=0; i<N_MAX; i++) {
                printf("Sin(%d) = %g\n", angle_arr[i], tay_arr[i]);
            }
            // Printing out the values calculated from the sine function
            printf("The actual sin(x) values are:\n");
            for (int i=0; i<N_MAX; i++) {
                printf("Sin(%d) = %g\n", angle_arr[i], sin_arr[i]);
            }
        } else {
            // If the number of terms is higher than what can be used in the function, an error message is displayed
            printf("The maximum number of terms allowed is 17! The Taylor series has not been calculated.\n");
        }
    }
    // If option 3 is chosen, every value is calculated and printed for both sin(x) and the Taylor series
    else if (option == '3') {
        printf("\nPrinting values from the both sin(x) and its Taylor series between 0 and 360 degrees to 6 terms.\n");
        printf("Values of sin(x) between 0 and 360 are:\n");
        // Calculating values for sin(x) between 0 and 360 degrees
        for (int i=0; i<=360; i++) {
            printf("Sin(%d) = %g\n", i, sin(i*(PI/180)));
        }
        printf("\nValues of the Taylor series of sin(x) between 0 and 360 are:\n");
        // Calculating values for the Taylor series between 0 and 360 degrees
        for (int i=0; i<=360; i++) {
            printf("Taylor sin(%d) = %g\n", i, sum_sin_series(i,6));
        }
    // Writing the values from the sin(x) and Taylor series functions to a new text file
    FILE *fptr;
    fptr = fopen("Sine and Taylor.txt", "w");
    if (fptr == NULL) {
        // Error message if the file doesn't open
        printf("Error opening file to write.\n");
        return 1;
    } else {
        // If the file opens successfully, the values of sin(x) and the Taylor series are written to the file
        for (int i=0; i<=360; i++) {
            fprintf(fptr, "%g, ", sin(i*(PI/180)));
        }
        for (int i=0; i<=360; i++) {
            fprintf(fptr, "%g, ", sum_sin_series(i,6));
        }
        // A confirmation is printed once the files have been written to the file
        printf("\nThese values have been written to a text file called 'Sine and Taylor.txt' for plotting.\n");
    }
    fclose(fptr);
    } else {
        // Ending the program if the user chooses to
        printf("Quitting program.\n");
        return 0;
    }
    return 0;
}

// Functions

int factorial(int n) {
    // Finds the value for the factorial of a positive integer given.
    // Float values are automatically truncated into integers.
    if (n>0) {
        int fact = 1;
        for (int i=0; i<n; i++) {
            fact *= n-i;
        }
        return fact;
    } else {
        printf("n must be an integer greater than 0.\n");
        return 1;
    }
}

double sum_sin_series(int x,int n) {
    // Calculates the Taylor series at an integer value x (in degrees), for sin(x) to the nth term specified.
    // Taking x and finding an equivalent point between -pi/2 and pi/2
    int angle;
    int new_angle;
    double radian;
    double taylor = 0;
    // Check if n is an allowed value
    if (n>0 && n<18) {
        // If n is allowed, check the angle entered is between -pi/2 and pi/2
        if (x>=-90 && x<=90) {
            radian = x*(PI/180);
        } else {
            // If the angle isn't in the range specified earlier, it's converted to an angle between -pi/2 and pi/2 radians that gives the same output.
            angle = x%360;
            if (angle>=-90 && angle<=90) {
                radian = angle*(PI/180);
            }
            else if (angle<=-90) {
                new_angle = -180-angle;
                radian = new_angle*(PI/180);
            } else {
                new_angle = 180-angle;
                radian = new_angle*(PI/180);
            }
        }
    } else {
        // Print an error message if n is an invalid value
        printf("n has to be an integer between 1 and 17!\n");
        return 1;
    }
    // Calculate the taylor series for sin(x)
    for (int i=1; i<n+1; i++) {
        int num = (2*i-1);
        taylor += pow(-1,i-1)*(pow(radian,num)/factorial(num));
    }
    return taylor;
}


