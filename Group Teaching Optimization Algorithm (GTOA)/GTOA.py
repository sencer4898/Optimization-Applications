# The problem has 10 design parameters.
# Each parameter can take values between 0.1 and 35.

import random
from trussBench import *
from solveTrussL import *
import copy
import numpy as np



# Create a population of solutions with randomly created design parameters.
def initPop(nPop):
    pop = []
    for i in range(nPop):
        # Each solution has 10 design parameters, 2 penalty values
        # and the weight of the setup.
        pop.append([random.uniform(0.1, 35) for i in range(10)]+[0, 0, 0])
    return pop


# Calculates the penalty parameters and weights of the population.
def evalPop(pop):
    newPop = copy.deepcopy(pop)
    for p in newPop:

        # Values beyond the parameter boundaries are set to the respective boundary. (Otherwise, evaluation
        # functions returns an error.)
        for i in range(10):
            if p[i] < 0.1:
                p[i] = 0.1
            elif p[i] > 35:
                p[i] = 35

        s10bar = copy.deepcopy(size10bar2D)
        s10bar[3] = p[:-3]
        sol = solveTrussL(s10bar)

        u = sol[1][0][1]
        s = sol[1][0][3]

        penD = 0
        for i in u:
            if abs(i[0]) > 2:
                penD += abs(i[0]) / 2 - 1
            if abs(i[1]) > 2:
                penD += abs(i[1]) / 2 - 1

        penS = 0
        for i in s:
            if abs(i) > 25:
                penS += abs(i) / 25 - 1

        p[-1] = sol[0]
        p[-2] = penD
        p[-3] = penS
    return (newPop)

# Returns the best solution of the population.
def best(pop):
    return(min(pop,key=lambda x:[x[-2]+x[-3],x[-1]]))  # Best solution is the solution with the least value of
# sum of penalty parameters. If there is a tie, solution with the least weight is selected.

# Returns the better of the two solutions.
def better(cand1,cand2):
    if (cand1[-2]+cand1[-3])<(cand2[-2]+cand2[-3]):
        return(cand1)
    elif (cand1[-2]+cand1[-3])>(cand2[-2]+cand2[-3]):
        return(cand2)
    elif cand1[-1]<cand2[-1]:
        return(cand1)
    elif cand1[-1]>=cand2[-1]:
        return(cand2)

# Selects the three best solution of the population. (For the Grey Wolf Optimizer (GWO) teacher selection)
def bestThree(pop):
    temp_pop = copy.deepcopy(pop)
    first = min(temp_pop,key=lambda x:[x[-2]+x[-3],x[-1]])  # Best solution
    temp_pop.remove(first)  # Remove the best solution
    second = min(temp_pop, key=lambda x: [x[-2] + x[-3], x[-1]])  # Second Best Solution
    temp_pop.remove(second)  # Remove the second best solution
    third = min(temp_pop, key=lambda x: [x[-2] + x[-3], x[-1]])  # Third Best Solution

    return [first, second, third]  # Returns the three best solutions as a list

# Takes the three best solution and returns the teacher
def selectTeacher(topThree):

    bestPop = topThree[0]  # En iyi çözüm
    tempThree = [(x + y + z)/3 for x, y, z in zip(topThree[0], topThree[1], topThree[2])]  # Middle point of
    # the three best solutions
    topThree = evalPop([tempThree])[0]  # Calculates the penalties and weight of the middle point solution

    return better(bestPop, topThree)  # Returns the best option as the teacher (Middle point or the first best solution)

# Divides the population into two: Average(Bad) and Outstanding(Good) populations
def groupPop(pop):

    temp_pop = copy.deepcopy(pop)
    goodPop = []

    # Select the good solutions
    for i in range(int(len(pop)/2)):
        current = min(temp_pop, key=lambda x:[x[-2]+x[-3],x[-1]])
        goodPop.append(current)
        temp_pop.remove(current)

    # Remaining solutions are assigned to the bad population
    badPop = temp_pop

    return badPop, goodPop  # Returns the bad and good populations


# Teacher and Student phases for the bad population
def teacherStudentPhaseBad(badPop, teacher):

    tempBad = []   # List to store the solutions created at the Teacher Phase
    newBad = []  # List to store the solutions created at the Student Phase

    # Teacher Phase

    for sol_ in badPop:
        tempSol = np.array(sol_) + 2 * random.uniform(0, 1) * (np.array(teacher) - np.array(sol_))  # Calculation
        #of the candidate solution according to the formula defined in the paper
        tempSol = evalPop([list(tempSol)])[0]  # Evaluation of the candidate solution (penalties, weight)
        tempBad.append(better(sol_, tempSol))  # Select the better of the candidate solution and the old solution

    # Student Phase

    for i in range(len(badPop)):  # Loop for all the solutions in the bad population

        e = random.uniform(0,1)
        g = random.uniform(0,1)

        j = random.randint(0, len(badPop)-1)  # Randomly select a solution (student)

        if better(tempBad[i],tempBad[j]) == tempBad[i]:  # If our solution is better than the randomly selected one
            newSol = np.array(tempBad[i]) + e * (np.array(tempBad[i]) - np.array(tempBad[j])) \
                     + g * (np.array(tempBad[i]) - np.array(badPop[i]))
            newSol = list(newSol)
        else:  # If randomly selected solution is better than our solution
            newSol = np.array(tempBad[i]) - e * (np.array(tempBad[i]) - np.array(tempBad[j])) \
                     + g * (np.array(tempBad[i]) - np.array(badPop[i]))
            newSol = list(newSol)

        newSol = evalPop([newSol])[0]  # Calculations of the penalties and the weight of the solution obtained
        # from the student phase

        newBad.append(better(newSol, tempBad[i]))  # Select the better of the pre-student phase and post-student phase
        #  solutions.

    return newBad  # Return the bad population trained in the student and teacher phases.

# Teacher and Student phases for the good population
def teacherStudentPhaseGood(goodPop, teacher):

    tempGood = []  # List to store the solutions created at the Teacher Phase
    newGood = []  # List to store the solutions created at the Student Phase
    mean = np.mean(goodPop, axis=0)  # Middle point solution for the good population
    mean = evalPop([list(mean)])[0]  # Evaluation of the middle point solution

    # Teacher Phase

    for sol_ in goodPop:
        b = random.uniform(0, 1)
        c = 1 - b
        F = random.randint(1, 2)
        tempGood.append(evalPop([list(np.array(sol_) + random.uniform(0, 1) * (np.array(teacher) \
                                            - F * (b * np.array(mean) + c * np.array(sol_))))])[0])

    # Student Phase

    for i in range(len(goodPop)):  # Loop for all the solutions in the good population

        e = random.uniform(0, 1)
        g = random.uniform(0, 1)

        j = random.randint(0, len(goodPop)-1)  # Randomly select a solution (student)

        if better(tempGood[i], tempGood[j]) == tempGood[i]:   # If our solution is better than the randomly selected one
            newSol = np.array(tempGood[i]) + e * (np.array(tempGood[i]) - np.array(tempGood[j])) \
                     + g * (np.array(tempGood[i]) - np.array(goodPop[i]))
            newSol = list(newSol)
        else:  # If randomly selected solution is better than our solution
            newSol = np.array(tempGood[i]) - e * (np.array(tempGood[i]) - np.array(tempGood[j])) \
                     + g * (np.array(tempGood[i]) - np.array(goodPop[i]))
            newSol = list(newSol)

        newSol = evalPop([newSol])[0]  # Calculations of the penalties and the weight of the solution obtained
        # from the student phase
        newGood.append(better(newSol, tempGood[i]))  # Select the better of the pre-student phase and post-student phase
        #  solutions.

    return newGood  # Return the good population trained in the student and teacher phases.


# Main Part of The Algorith

# ---------------------------
nPop = 50  # Population Size (N)
tCurrent = 0  # Current number of function evaluations
tMax = 50000  # Maximum number of function evaluations
# ---------------------------

pop = initPop(nPop)  # Initiate the population

print("0. İterasyon --- ", best(evalPop(pop))[-3:])  # Print the initial best solution

for k in range(1, tMax):  # Run the algorithm until the stopping criteria is satisfied.
    if tCurrent > tMax:  # Check the stopping criteria
        break
    pop = evalPop(pop) # Evaluate the current population
    tCurrent += nPop  # Increase the current number of function evaluations by population size (N)

    teacher = selectTeacher(bestThree(pop))  # Teacher assignment

    badPop, goodPop = groupPop(pop)  # Divide the population into two (Good and Bad)
    badPop = teacherStudentPhaseBad(badPop, teacher)  # Teacher and Student phases for bad population
    goodPop = teacherStudentPhaseGood(goodPop, teacher)  # Teacher and Student phases for good population

    pop = badPop + goodPop  # Merge the populations

    tCurrent += 2*len(pop) + 1  # Increase the current number of function evaluations by 2N+1

    print(k, ". İterasyon --- ", best(pop)[-3:])  # Print the penalty parameters
                                                  # and the weight of the current best solution.

# Print the final solution
print("-------------------")
print("en iyi çözüm")
print(best(pop)[:-3])
print(best(pop)[-3:])

# Additional Comments:

# Without a large population size and a lot of iterations, optimality with this algorithm is unlikely to be achieved.
# GTOA might not be convinient for this problem at hand.