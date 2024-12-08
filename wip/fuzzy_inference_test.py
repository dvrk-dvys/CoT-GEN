#https://www.tiktok.com/@_gcanale/photo/7374174046830808353?lang=en
'''
Fuzzy logic is a computational approach that deals with the approximate reasoning and imprecise information.
It allows for defrees of truth or membership, unlike traditiojnal binary logic where a statement is either
true or false. Fuzzy logic is particularly useful in scenarios where information is incomlpete, ambiguous,
 or imprecise, making it a powerful too for decision-making and control systems.

In fuzzy logic, the fundamental concept is the fuzy set. A fuzzy set is where elements can have varying
degrees of membership, represented by a membership function. The membership function maps each element
to a value between 0 and 1m indicating the degree of which the element belongs to the set.
This is in contrast to classical sets, where an element either belongs to the set (membership value of 1)
or does not (membership value of 0).
'''



import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define a fuzzy set for temperature
temperature = np.arange(0, 101, 1)
cold = np.fmax(0, (20 - temperature) / 20)
hot = np.fmax(0, (temperature - 80) / 20)


# Print the membership values for a few temperatures
print(f"Membership for 10°C: Cold = {cold[10]}, Hot = {hot[10]}")
print(f"Membership for 50°C: Cold = {cold[50]}, Hot = {hot[50]}")
print(f"Membership for 90°C: Cold = {cold[90]}, Hot = {hot[90]}")

'''
#Fuzzy Operations
#Fuzzy sets support various operations similar to classical set operations, such as union, intersection, and compliment.
#However, these operations are defined differently in Fuzzy logicm, acounting for the varying degrees of membership.
#The union operation takes the maximum value, the intersection operations takes the minimum membership value,
# and the cimplement oepration subtract the membership value from 1.
'''

# def fuzzy sets
a = np.array([0.2, 0.5, 0.7, 1.0])
b = np.array([0.1, 0.3, 0.9, 0.6])

#Fuzzy union
c = np.fmax(a, b)
print(f"Fuzzy Union:", c)

#Fuzzy intersection
d = np.fmin(a, b)
print(f"Fuzzy Intersection:", d)

#Fuzzy complement
e = 1 - a
print(f"Fuzzy complement:", e)

'''
Fuzzy rules and inference systems
Fuzzy logic employs fuzzy rules and inference syystems to make dcisions or sontrol processebased on fuzzy inputs and outputs.
A fuzzy rule tales form 'IF' (antecedent) THEN (consequent)," where both the antecedent and consequent are fuzzy statements.
These rules are combined using ann inference system, such as Mamdani or Sugeno method, to produce a fuzzy output. 
'''


# Define input variables
temperature = np.arange(0, 101, 1)
cold = fuzz.membership.trimf(temperature, [0, 0, 20])
hot = fuzz.membership.trimf(temperature, [80, 100, 100])

#define output variable
fan_speed = np.arange(0, 11, 1)
low = fuzz.membership.trimf(fan_speed, [0, 0, 5])
high = fuzz.membership.trimf(fan_speed, [5, 10, 10])

# define fuzzy rules
rule1 = np.fmin(cold, low) # IF cold THEN low fan speed
rule2 = np.fmin(hot, high) # IF hot THEN high fan speed

#Aggregate rules
aggregated = np.fmax(rule1, rule2)

#Defuzzify output
fan_level = fuzz.defuzz(fan_speed, aggregated, 'centriod')
print(f"fan speed should be: {fan_level}")

'''
FUZZIFICATION AND DEFUZZIFICATION
Fuzzificaiton is the preocess of converting crisp input values into fuzzy sets, while defuzzification is the process of converting
the fuzzy output sets into crisp values. 
Fuzzificatiion involves mapping input values to their corresponding membership values in the defined fuzzy sets. 
Defuzzification utilizes methods like the centroid, mean of maximum, or smallest of maximum to derive a single output value from the aggregated fuzzy output.
'''

# Define input and output variables
distance = np.arange(0, 11, 1)
speed = np.arange(0, 101, 1)

#Fuzzify input
near = fuzz.membership.trimf(distance, [0, 0, 5])
far = fuzz.membership.trimf(distance, [5, 10, 10])

#Defuzzify input
slow = fuzz.membership.trimf(distance, [0, 0, 50])
fast = fuzz.membership.trimf(distance, [5, 100, 100])

#Apply fuzzy rules
rule1 = np.fmin(near, slow)
rule2 = np.fmin(far, fast)
aggregated = np.fmax(rule1, rule2)

# Defuzzify output
crisp_speed = fuzz.defuzz(speed, aggregated, 'centroid')
print(f"Recommended speed: {crisp_speed} km/h")



"""
Fuzzy Control Systems
Fuzzy Control Systems are widely used in various applications, such as industrial process control, consumer electronics and robotics.
They employ  Fuzzy logic to output variables through a set of fuzzy rules and inference mechanisms. 
These systems can handle inprecise or vaue input to provide smooth, adaptive control outputs. 
"""

# Define input variables
temperature = np.arange(0, 101, 1)
humidity = np.arange(0, 101, 1)

# Define output variables
air_conditioning = np.arange(0, 11, 1)

#fuzzify input variables
cold = fuzz.membership.trimf(temperature, [0, 0, 20])
warm = fuzz.membership.trimf(temperature, [15, 25, 35])
hot = fuzz.membership.trimf(temperature, [30, 40, 100])

dry = fuzz.membership.trimf(humidity, [0, 0, 40])
comfortable = fuzz.membership.trimf(humidity, [30, 50, 70])
humid = fuzz.membership.trimf(humidity, [60, 80, 100])

#define output membership function
low_ac = fuzz.membership.trimf(air_conditioning, [0, 0, 4])
medium_ac = fuzz.membership.trimf(air_conditioning, [2, 5, 8])
high_ac = fuzz.membership.trimf(air_conditioning, [6, 10, 10])

# define fuzzy rules
rule1 = np.fmax(cold, dry) # IF cool THEN dry
rule2 = np.fmin(warm, comfortable) # IF warm THEN comfortable
rule3 = np.fmin(hot, humid) # IF hot THEN humid

#? why no aggregation here?

#Defuzzify output
ac_level = fuzz.defuzz(air_conditioning, np.fmax(rule1, np.fmax(not rule2, rule3)), 'centriod')
print(f"Air Conditioning: {ac_level}")

'''
Fuzzy Logic Toolbox in Python
Python provides several libraries and toolboxes for working with fuzzy logic, making it easier to implement fuzzy systems and applications. 
One popular library is scikit-fuzzy, which offers a comprehensive set of functions and tools for buulding and analyzing fuzzy systems,
 including membership functions, fuzzy operations, and inference mechanisms.
'''

# Define input variables
distance = np.arange(0, 11, 1)

#define membership functions
near = fuzz.membership.trimf(distance, [0, 0, 5])
far = fuzz.membership.trimf(distance, [5, 10, 10])

#Visualize membership functions
plt.figure()
plt.plot(distance, near, 'r', linewidth=1.5, label='Near')
plt.plot(distance, far, 'b', linewidth=1.5, label='Far')
plt.legend()
plt.title('Distance Membership Functions')
plt.show()
