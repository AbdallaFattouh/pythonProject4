from lcoe.lcoe import lcoe
import numpy as np
from numpy_financial import irr
from numpy_financial import npv
import matplotlib.pyplot as plt

operating_cost = 1000  # $million/year
capital_cost = 50000  # $million
discount_rate = 0.02  # %
lifetime = 25
annual_output = 25000  # kWh
lcoe_value = lcoe(annual_output, capital_cost, operating_cost, discount_rate, lifetime)
print("lcoe is:", lcoe_value)
if lcoe_value < 0.5:
    print("profits per kWh:", 0.5 - lcoe_value)
else:
    print("lcoe is bb above market price")
lcoe_profit: float = (0.5 - lcoe_value) * annual_output
y = 1 + discount_rate
print("annual revenue is:", lcoe_profit)

irr_input = np.zeros(25)
irr_input[0] = - capital_cost

for j in range(1, 25):
    irr_input[j] = lcoe_profit/y**j

irr_output = irr(irr_input)

print("irr is:", irr_output)

cashflows = [-500, 200, 147, 128, 130, 235]  # t0, t1, t2, t3, t4, t5

discountRate = 0.9  # Nine percent per annum

npv = np.npv(discountRate, cashflows)

print("Net present value of the investment:%3.2f" % npv)

plt.plot([lcoe_value], [irr_output], 'ro')
#plt.show()