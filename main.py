from lcoe.lcoe import lcoe
import numpy as np
from numpy_financial import irr
import matplotlib.pyplot as plt

operating_cost = 1000  # $million/year
capital_cost = 50000  # $million
discount_rate = 0.02  # %
lifetime = 25
annual_output = 25000  # kWh

capital_cost = [50, 100, 200, 400]
operating_costs = [5, 10, 20, 40]


costs = [capital_cost, operating_costs, discount_rate]

for cost in costs:
    list_lcoe = []
    list_irr = []
    #figure = 1
    for n in cost:

        lcoe_value = lcoe(annual_output, n, operating_cost, discount_rate, lifetime)
        print("lcoe is:", lcoe_value)
        if lcoe_value < 0.5:
            print("profits per kWh:", 0.5 - lcoe_value)
        else:
            print("lcoe is above market price")
        lcoe_profit: float = (0.5 - lcoe_value) * annual_output
        y = 1 + discount_rate
        print("annual revenue is:", lcoe_profit)

        irr_input = np.zeros(25)
        irr_input[0] = - n

        for j in range(1, 25):
            irr_input[j] = lcoe_profit/y**j

        irr_output = irr(irr_input)

        print("irr is:", irr_output)

        list_lcoe.append(lcoe_value)
        list_irr.append(irr_output)



    #plt.plot([capital_cost], [irr_output], 'ro')
    #plt.plot([50000, 100000, 200000, 40000], [0.14244087683478937, 0.24488175366957873, 0.44976350733915743, 0.12195270146783148], 'ro')
    #plt.ylabel('lcoe')
    #plt.xlabel('Capital Cost')
    #plt.show()
    print(list_lcoe)
    print(list_irr)
    plot = plt.figure()
    plt.plot(cost, list_irr)
    #figure = figure + 1
    plt.show()

"""
class Test():

    def funct1(self):
        pass
    def func2(self):
        pass
"""