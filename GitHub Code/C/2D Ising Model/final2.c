/*
 * Final Project for PHY2027
 * Author: Elijah Gill
 * Due Date: 15/12/2023
*/

/*
 * Program description: This program simulates a 2D, 2-spin Ising model. It includes 4 options
 *                      to make random lattices and test the functions to calculate their energies
 *                      and normalised magnetisations. The latter 2 options are for writing data to
 *                      text files in order to visualise the data through graphs.
 * 
 *                      This was the final project for my 'Programming in C' module in my 2nd year of university.
 *                      I scored 75% on this project.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define NUM_ROWS 25
#define NUM_COLS 25

const double kb_const = 1.38065e-23;
// Common terms are asked for in functions so are made into global variables
float J; // Interaction term
float h; // External  magnetic field term
float mu; // Magnetic moment term

// Declaring functions
void user_input(void);
float magnetisation(int *lattice);
int site_spin(int *array,int i,int j);
int adjacent_left(int *array,int i,int j);
int adjacent_right(int *array,int i,int j);
int adjacent_up(int *array,int i,int j);
int adjacent_down(int *array,int i,int j);
float energyH(int *array, int i, int j, float J, float h, float mu);
float total_energy(int *array, float J, float h, float mu);
float energy_change(int *array, int i, int j, float J, float h, float mu);
int random_i(void);
int random_j(void);
void prob_flip(int *array, int i, int j, float J, float h, float mu, float temp);
void prob_flip_beta(int *array, int i, int j, float J, float h, float mu, float beta);

int main() {

    // Variables
    char option; // Used in the menu to check the entered character
    float temp; // Temperature
    int evolution; // Number of evolutions the lattice goes through

    // Making a menu for the user to choose what to do
    printf("==================================================\n");
    printf("=========== Final Project: Ising Model ===========\n");
    printf("==================================================\n");
    printf(" --- Please choose an option ---\n");
     // Giving the user 3 different options to choose, or to quit the program instead
    do {
        printf(" - Enter '1' to create a random lattice and find its total energy and normalised magnetisation.\n");
        printf(" - Enter '2' to simulate the evolution of the lattice.\n");
        printf(" - Enter '3' to obtain values for the normalised magnetisation depending on the magnetic field and write them to a text file.\n");
        printf(" - Enter '4' to write the evolution of the lattice to a text file.\n");
        printf(" - Enter 'q' to quit the program.\n");
        scanf(" %c", &option);
        while (getchar() != '\n');
        if ((option != '1') && (option != '2') && (option != '3') && (option != '4') && (option != 'q')) {
            printf("Please choose an available option.\n\n");
        }
    }
    while ((option != '1') && (option != '2') && (option != '3') && (option != '4') && (option != 'q'));

    // If option 1 is chosen, the user is asked to input values for the interaction, magnetic field and magnetic moment terms
    if (option == '1') {
        printf("\nYou chose option 1!\n\n");
        user_input();
        printf("Generating a random lattice:\n\n");
        // Making the randomly generated lattice
        time_t t; // Stores the time when the code is ran
        srand((unsigned)time(&t)); //Random number generation seed
        //Allocate memory for the lattice
        int* lattice = (int*)malloc(NUM_ROWS*NUM_COLS*sizeof(int));
        // Check if the memory is successfully allocated
        if (lattice == NULL) {
            fprintf(stderr, "Error - Memory not allocated\n");
            exit(1);
        }
         // Initialise lattice
        for(int i=0; i<NUM_ROWS*NUM_COLS; i++) {
            int num = rand()%2;
            if (num == 0) {
                num = -1;
            }
            lattice[i] = num;
        }

        //Print lattice
        for(int i=0; i<NUM_ROWS; i++) {
            for(int j=0; j<NUM_COLS; j++) {
                printf("%2d ", lattice[i*NUM_COLS+j]);
            } printf("\n");
        }
        // Printing the values for the total energy and normalised magnetisation of the lattice
        printf("\nInteraction (J) = %f, Magnetic Field (h) = %f, Magnetic Moment (mu) = %f\n", J, h, mu);
        printf("Total energy of the lattice = %f\n", total_energy(&lattice[0],J,h,mu));
        printf("Normalised magnetisation = %f\n", magnetisation(&lattice[0]));
        // Freeing the reserved memory used for the lattice
        free(lattice);
        lattice = NULL;
    }

    // If option 2 is chosen, the user is asked the same prompts as option 1 as well as the number of times to evolve
    if (option == '2') {
        printf("\nYou chose option 2!\n\n");
        user_input();
        printf("Please enter a value for the temperature.\n");
        scanf("%f", &temp);
        printf("Please enter the number of evolutions for the lattice.\n");
        scanf("%d", &evolution);
        printf("Generating a random lattice:\n\n");
        // Making the randomly generated lattice
        time_t t; // Stores the time when the code is ran
        srand((unsigned)time(&t)); //Random number generation seed
        //Allocate memory for the lattice
        int* lattice = (int*)malloc(NUM_ROWS*NUM_COLS*sizeof(int));
        // Check if the memory is successfully allocated
        if (lattice == NULL) {
            fprintf(stderr, "Error - Memory not allocated\n");
            exit(1);
        }
         // Initialise lattice
        for(int i=0; i<NUM_ROWS*NUM_COLS; i++) {
            int num = rand()%2;
            if (num == 0) {
                num = -1;
            }
            lattice[i] = num;
        }

        //Print lattice
        for(int i=0; i<NUM_ROWS; i++) {
            for(int j=0; j<NUM_COLS; j++) {
                printf("%2d ", lattice[i*NUM_COLS+j]);
            } printf("\n");
        }
        // Printing the values for the total energy and normalised magnetisation of the lattice
        printf("\nInteraction (J) = %f, Magnetic Field (h) = %f, Magnetic Moment (mu) = %f, Evolutions = %d\n", J, h, mu, evolution);
        printf("Total energy of the lattice = %f\n", total_energy(&lattice[0],J,h,mu));
        printf("Normalised magnetisation = %f\n", magnetisation(&lattice[0]));
        // Run lattice for the number of evolutions
        for (int i=0; i<evolution; i++) {
            // Generating random sites
            int site_i = random_i();
            int site_j = random_j();
            // Running the probability to flip spin function
            prob_flip(&lattice[0],site_i,site_j,J,h,mu,temp);
        }
        printf("\n");
        //Print new lattice
        for(int i=0; i<NUM_ROWS; i++) {
            for(int j=0; j<NUM_COLS; j++) {
                printf("%2d ", lattice[i*NUM_COLS+j]);
            } printf("\n");
        }
        // Printing the new values for the total energy and normalised magnetisation of the lattice
        printf("\nInteraction (J) = %f, Magnetic Field (h) = %f, Magnetic Moment (mu) = %f, Evolutions = %d\n", J, h, mu, evolution);
        printf("Total energy of the lattice = %f\n", total_energy(&lattice[0],J,h,mu));
        printf("Normalised magnetisation = %f\n", magnetisation(&lattice[0]));
        // Freeing the reserved memory used for the lattice
        free(lattice);
        lattice = NULL;
    }

    // If option 3 is chosen, lattices with differing values for h are evolved and their magnetisations are written to a text file
    // This was used the generate the plots in the pdf explanation of the code
    if (option == '3') {
        printf("\nYou chose option 3!\n\n");
        // Asking for user input
        printf("Please enter a value for the interaction term.\n");
        scanf("%f", &J);
        printf("Please enter a value for the magnetic moment.\n");
        scanf("%f", &mu);
        printf("Please enter a value for beta (proportional to 1/temperature).\n");
        scanf("%f", &temp);
        printf("Please enter the number of evolutions for the lattice.\n");
        scanf("%d", &evolution);
        // Printing the inputs so they can be checked by the user
        printf("\nInteraction (J) = %f, Beta = %f, Magnetic Moment (mu) = %f, Evolutions = %d\n", J, temp, mu, evolution);
        // Write the calculated data to a file
        FILE *fptr;
        fptr = fopen("Norm_Mag v Magnet_B.txt", "w");
        if (fptr == NULL) {
            // Error message if the file doesn't open
            printf("Error opening file to write.\n");
            return 1;
        } else {
            // Write the values for the interaction, magnetisation moment and temperature
            fprintf(fptr, "%f, %f, %f, ", J, mu, temp);
            // for loop to generate data
            for (int i=-50; i<=50; i++) {
                float h = (float)i/10; // h has a value from -5 to 5 with a step of 0.1
                // Generating the same initial lattice each time
                // Allocating memory for the lattice
                int* lattice = (int*)malloc(NUM_ROWS*NUM_COLS*sizeof(int));
                // Check if the memory is successfully allocated
                if (lattice == NULL) {
                    fprintf(stderr, "Error - Memory not allocated\n");
                    exit(1);
                }
                // Initialise lattice
                for(int i=0; i<NUM_ROWS*NUM_COLS; i++) {
                    int num = rand()%2;
                    if (num == 0) {
                        num = -1;
                    }
                    lattice[i] = num;
                }
                // Run each lattice for a number of evolutions
                for (int i=0; i<evolution; i++) {
                    // Generating random sites
                    int site_i = random_i();
                    int site_j = random_j();
                    // Running the probability to flip spin function
                    prob_flip_beta(&lattice[0],site_i,site_j,J,h,mu,temp);
                }
                // Calculate the normalised magnetisation
                float norm_mag = magnetisation(&lattice[0]);
                // Write this value to the text file
                fprintf(fptr, "%f, ", norm_mag);
                // Freeing the reserved memory used for the lattice
                free(lattice);
                lattice = NULL;
            }
        }
        // A confirmation is printed once the files have been written to the file
        printf("\nThese values have been written to a text file called 'Norm_Mag v Magnet_B.txt'.\n");
        fclose(fptr);
    }

    // If option 4 is chosen, every evolution of the lattice is written to a text file
    // This was used the generate the plots in the pdf explanation of the code
    if (option == '4') {
        printf("\nYou chose option 4!\n\n");
        // Asking for user input
        user_input();
        printf("Please enter a value for the temperature.\n");
        scanf("%f", &temp);
        printf("Please enter the number of evolutions for the lattice.\n");
        scanf("%d", &evolution);
        // Printing the inputs so they can be checked by the user
        printf("\nInteraction (J) = %f, Temp = %f, Magnetic Moment (mu) = %f, Evolutions = %d\n", J, temp, mu, evolution);
        // Making the randomly generated lattice
        time_t t; // Stores the time when the code is ran
        srand((unsigned)time(&t)); //Random number generation seed
        //Allocate memory for the lattice
        int* lattice = (int*)malloc(NUM_ROWS*NUM_COLS*sizeof(int));
        // Check if the memory is successfully allocated
        if (lattice == NULL) {
            fprintf(stderr, "Error - Memory not allocated\n");
            exit(1);
        }
         // Initialise lattice
        for(int i=0; i<NUM_ROWS*NUM_COLS; i++) {
            int num = rand()%2;
            if (num == 0) {
                num = -1;
            }
            lattice[i] = num;
        }

        // Printing the lattice after each iteration to a text file
        FILE *fptr2;
        fptr2 = fopen("Lattice_Spin_Data_hightemp.txt", "w");
        if (fptr2 == NULL) {
            // Error message if the file doesn't open
            printf("Error opening file to write.\n");
            return 1;
        } else {
            // Write the lattice to a text file
            for(int i=0; i<NUM_ROWS; i++) {
                for(int j=0; j<NUM_COLS; j++) {
                    fprintf(fptr2, "%d,", lattice[i*NUM_COLS+j]);
                } fprintf(fptr2, "\n");
            }
            // For each evolution, write to the txt file
            for (int i=0; i<evolution; i++) {
                // Generating random sites
                int site_i = random_i();
                int site_j = random_j();
                // Running the probability to flip spin function
                prob_flip(&lattice[0],site_i,site_j,J,h,mu,temp);
                // Write iteration to the txt file
                for(int i=0; i<NUM_ROWS; i++) {
                    for(int j=0; j<NUM_COLS; j++) {
                        fprintf(fptr2, "%d,", lattice[i*NUM_COLS+j]);
                    } fprintf(fptr2, "\n");
                }
            }
        }
        // A confirmation is printed once the files have been written to the file
        printf("\nValues successfully written to 'Lattice_Spin_Data.txt'.\n");
        fclose(fptr2);
        // Freeing the reserved memory used for the lattice
        free(lattice);
        lattice = NULL;
    }
    if (option == 'q') {
        // Ending the program if the user chooses to
        printf("\nQuitting program...\n");
        return 0;
    }
    return 0;
}

// Functions

void user_input(void) {
    // Asks the user to input values for the interaction, magnetic field and magnetic moment terms
    printf("Please enter a value for the interaction term.\n");
    scanf("%f", &J);
    printf("Please enter a value for the external magnetic field term.\n");
    scanf("%f", &h);
    printf("Please enter a value for the magnetic moment.\n");
    scanf("%f", &mu);
}

float magnetisation(int *lattice) {
    // Finds the average spin (magnetisation) of the lattice
    int sum_spin = 0;
    for (int i=0; i<NUM_COLS; i++) {
        for (int j=0; j<NUM_ROWS; j++) {
            // Finding the given site
            int *site = lattice + (i*NUM_COLS+j);
            sum_spin += *site;
        }
    }
    float average_spin = (float)sum_spin / (NUM_COLS*NUM_ROWS);
    return average_spin;
}

// Functions related to indexing the lattice

int site_spin(int *array,int i,int j) {
    // Finding the given site
    int *site = array + (i*NUM_COLS+j);
    return *site;
}
// Functions to return the spins of the nearest neighbours
int adjacent_left(int *array,int i,int j) {
    // Finding the given site
    int *site = array + (i*NUM_COLS+j);
    // Checking if the site is on the left edge of the lattice
    if (((i*NUM_COLS+j)%NUM_COLS)==0) {
        return *(site+NUM_COLS-1);
    }
    // Finding the left nearest neighbour
    int *leftAdj = site - 1;
    return *leftAdj;
}
int adjacent_right(int *array,int i,int j) {
    // Finding the given site
    int *site = array + (i*NUM_COLS+j);
    // Checking if the site is on the right edge of the lattice
    if (((i*NUM_COLS+j+1)%NUM_COLS)==0) {
        return *(site-NUM_COLS+1);
    }
    // Finding the right nearest neighbour
    int *rightAdj = site + 1;
    return *rightAdj;
}
int adjacent_up(int *array,int i,int j) {
    // Finding the given site
    int *site = array + (i*NUM_COLS+j);
    // Checking if the site is on the top edge of the lattice
    if ((i*NUM_COLS+j-NUM_COLS)<0) {
        return *(site+NUM_COLS*(NUM_ROWS-1));
    }
    // Finding the nearest neighbour above the site
    int *upAdj = site - NUM_COLS;
    return *upAdj;
}
int adjacent_down(int *array,int i,int j) {
    // Finding the given site
    int *site = array + (i*NUM_COLS+j);
    // Checking if the site is on the bottom edge of the lattice
    if ((i*NUM_COLS+j+NUM_COLS)>=(NUM_ROWS*NUM_COLS)) {
        return *(site-NUM_COLS*(NUM_ROWS-1));
    }
    // Finding the nearest neighbour below the site
    int *downAdj = site + NUM_COLS;
    return *downAdj;
}

// Functions relating to energy

float energyH(int *array, int i, int j, float J, float h, float mu) {
    // Function to find the energy at a single site in the lattice
    // Finding the given site
    int *site = array + (i*NUM_COLS+j);
    // Finding the spin of nearest neighbours
    int left_spin = adjacent_left(array,i,j);
    int right_spin = adjacent_right(array,i,j);
    int up_spin = adjacent_up(array,i,j);
    int down_spin = adjacent_down(array,i,j);
    // First summation
    float sum1 = 0;
    sum1 = J*(*site)*(left_spin + right_spin + up_spin + down_spin);
    // Second summation
    float sum2 = 0;
    sum2 = mu*h*((*site)+left_spin + right_spin + up_spin + down_spin);
    return -sum1 - sum2;
}

float total_energy(int *array, float J, float h, float mu) {
    // Function to find the total energy in the lattice
    float sum_E = 0;
    for (int i=0; i<NUM_COLS; i++) {
        for (int j=0; j<NUM_ROWS; j++) {
            sum_E += energyH(array,i,j,J,h,mu);
        }
    }
    return sum_E;
}

float energy_change(int *array, int i, int j, float J, float h, float mu) {
    // Function to find the change in energy if the spin is flipped
    // Finding the given site
    int *site = array + (i*NUM_COLS+j);
    // Finding the spin of nearest neighbours
    int left_spin = adjacent_left(array,i,j);
    int right_spin = adjacent_right(array,i,j);
    int up_spin = adjacent_up(array,i,j);
    int down_spin = adjacent_down(array,i,j);
    // Call energy function for current spin
    float current_energy = energyH(array,i,j,J,h,mu);
    // Change the spin of the site
    int site_tmp;
    if (*site==1) {
        site_tmp = -1;
    } else {
        site_tmp = 1;
    }
    // Find energy for changed spin
    // First summation
    float sum1 = 0;
    sum1 = J*(site_tmp)*(left_spin + right_spin + up_spin + down_spin);
    // Second summation
    float sum2 = 0;
    sum2 = mu*h*((site_tmp) + left_spin + right_spin + up_spin + down_spin);
    float energy_changed = -sum1 - sum2;
    // Return difference in energies
    return energy_changed - current_energy;
}

// Randomly generate i and j values for points in the lattice
int random_i(void) {
    return rand()%NUM_COLS;
}
int random_j(void) {
    return rand()%NUM_ROWS;
}

// Probability function
void prob_flip(int *array, int i, int j, float J, float h, float mu, float temp) {
    // Function flips the spin of the site if it meets the criterea
    // Check if the change in energy is positive or negative
    double prob;
    float diff_E = energy_change(array,i,j,J,h,mu);
    if (diff_E < 0) {
        prob = 1;
    } else {
        prob = exp(-diff_E/(kb_const*temp));
    }
    // Use a randomly generated number to simulate probability
    float rand_num = rand()%1000;
    rand_num = rand_num/1000;
    if (rand_num < prob) {
        // Finding the given site
        int *site = array + (i*NUM_COLS+j);
        // Changing the spin of the site
        if (*site == 1) {
            *site = -1;
        } else {
            *site = 1;
        }
    }
}

// Probability function using beta
void prob_flip_beta(int *array, int i, int j, float J, float h, float mu, float beta) {
    // Function flips the spin of the site if it meets the criterea
    // Check if the change in energy is positive or negative
    double prob;
    float diff_E = energy_change(array,i,j,J,h,mu);
    if (diff_E < 0) {
        prob = 1;
    } else {
        prob = exp(-diff_E*beta);
    }
    // Use a randomly generated number to simulate probability
    float rand_num = rand()%1000;
    rand_num = rand_num/1000;
    if (rand_num < prob) {
        // Finding the given site
        int *site = array + (i*NUM_COLS+j);
        // Changing the spin of the site
        if (*site == 1) {
            *site = -1;
        } else {
            *site = 1;
        }
    }
}
