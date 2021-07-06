import numpy as np
from numpy_financial import irr
import matplotlib.pyplot as plt
import h5py
import numpy_financial as npf
from numpy_financial import irr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean


def lcoe(total_cost, OM, lifetime, discount_rate, total_energy):
    # print(total_costs / df["total_energy"])

    lcoe_value = (((total_cost + OM * lifetime) / (1 + discount_rate) ** lifetime) / (
            total_energy * lifetime / (1 + discount_rate) ** lifetime))
    # print("lcoe is:", lcoe_value, "€/mWh")

    return lcoe_value


def irr(lifetime, total_cost, annual_profit):
    irr_input = np.zeros(lifetime)
    irr_input[0] = - total_cost

    for j in range(1, lifetime):
        irr_input[j] = annual_profit  # / y ** j

    irr_output = npf.irr(irr_input)

    # print("irr is:", irr_output)

    return irr_output


def npv(annual_profit, y, total_cost, lifetime):
    npv_input = np.zeros(lifetime)
    npv_input[0] = - total_cost
    for i in range(1, lifetime):
        npv_input[i] = annual_profit / y ** i
    npv_sum = 0
    for each in npv_input:
        npv_sum = npv_sum + each
    return npv_sum


# sensitivity analasys and plotting lcoe


def plot_variable_total_cost(total_costs):
    lcoe_values = []
    for total_cost in total_costs:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        lcoe_values.append(lcoe_value)
    print("lcoe for total cost", lcoe_values)
    # pd.DataFrame(lcoe_values).to_csv("lcoe total cost.csv")
    plot = plt.figure()
    plt.plot(total_costs, lcoe_values)
    plt.ylabel('lcoe')
    plt.xlabel('Capital Cost')
    plt.grid()
    plt.show()


def plot_variable_OM(OMs):
    lcoe_values = []
    for OM in OMs:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        lcoe_values.append(lcoe_value)
    print("lcoe for OM", lcoe_values)
    # pd.DataFrame(lcoe_values).to_csv("lcoe OM.csv")
    plot = plt.figure()
    plt.plot(OMs, lcoe_values)
    plt.ylabel('lcoe')
    plt.xlabel('OM')
    plt.grid()
    plt.show()


def plot_variable_discount_rate(discount_rates):
    lcoe_values = []
    for discount_rate in discount_rates:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        lcoe_values.append(lcoe_value)
    print("lcoe for discount rate", lcoe_values)
    plot = plt.figure()
    plt.plot(discount_rates, lcoe_values)
    plt.ylabel('lcoe')
    plt.xlabel('discount rate')
    plt.grid()
    plt.show()


def plot_variable_lifetimes(lifetimes):
    lcoe_values = []
    for lifetime in lifetimes:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        lcoe_values.append(lcoe_value)
    print("lcoe for lifetimes", lcoe_values)
    plot = plt.figure()
    plt.plot(lifetimes, lcoe_values)
    plt.ylabel('lcoe')
    plt.xlabel('lifetime')
    plt.grid()
    plt.show()


def plot_variable_total_energies(total_energies):
    lcoe_values = []
    for total_energy in total_energies:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        lcoe_values.append(lcoe_value)
    print("lcoe for total_energies", lcoe_values)
    plot = plt.figure()
    plt.plot(total_energies, lcoe_values)
    plt.ylabel('lcoe')
    plt.xlabel('total energy')
    plt.grid()
    plt.show()


# sensititvity analasiys and plotting irr

def plot_variable_irr_total_cost(total_costs):
    irr_values = []
    for total_cost in total_costs:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        # if lcoe_value < market_price:
        #    print("profits per mWh:", profits)
        # else:
        #    print("lcoe is above market price")
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_irr = irr(lifetime=lifetime, total_cost=total_cost, annual_profit=annual_profit)
        irr_values.append(my_irr)
    print("irr for total cost", irr_values)
    plot = plt.figure()
    plt.plot(total_costs, irr_values)
    plt.ylabel('irr')
    plt.xlabel('Capital Cost')
    plt.grid()
    plt.show()


def plot_variable_irr_OM(OMs):
    irr_values = []
    for OM in OMs:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        # if lcoe_value < market_price:
        #    print("profits per mWh:", profits)
        # else:
        #    print("lcoe is above market price")
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_irr = irr(lifetime=lifetime, total_cost=total_cost, annual_profit=annual_profit)
        irr_values.append(my_irr)
    print("irr for OM", irr_values)
    plot = plt.figure()
    plt.plot(OMs, irr_values)
    plt.ylabel('irr')
    plt.xlabel('OM')
    plt.grid()
    plt.show()


def plot_variable_irr_discount_rates(discount_rates):
    irr_values = []
    for discount_rate in discount_rates:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        # if lcoe_value < market_price:
        #    print("profits per mWh:", profits)
        # else:
        #    print("lcoe is above market price")
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_irr = irr(lifetime=lifetime, total_cost=total_cost, annual_profit=annual_profit)
        irr_values.append(my_irr)
    print("irr for discount rates", irr_values)
    plot = plt.figure()
    plt.plot(discount_rates, irr_values)
    plt.ylabel('irr')
    plt.xlabel('discount rate')
    plt.grid()
    plt.show()


def plot_variable_irr_lifetimes(lifetimes):
    irr_values = []
    for lifetime in lifetimes:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        # if lcoe_value < market_price:
        #    print("profits per mWh:", profits)
        # else:
        #    print("lcoe is above market price")
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_irr = irr(lifetime=lifetime, total_cost=total_cost, annual_profit=annual_profit)
        irr_values.append(my_irr)
    print("irr for lifetimes", irr_values)
    plot = plt.figure()
    plt.plot(lifetimes, irr_values)
    plt.ylabel('irr')
    plt.xlabel('lifetimes')
    plt.grid()
    plt.show()


def plot_variable_irr_total_energy(total_energies):
    irr_values = []
    for total_energy in total_energies:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        # if lcoe_value < market_price:
        #    print("profits per mWh:", profits)
        # else:
        #    print("lcoe is above market price")
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_irr = irr(lifetime=lifetime, total_cost=total_cost, annual_profit=annual_profit)
        irr_values.append(my_irr)
    print("irr for total energy", irr_values)
    plot = plt.figure()
    plt.plot(total_energies, irr_values)
    plt.ylabel('irr')
    plt.xlabel('total energy')
    plt.grid()
    plt.show()


# sensitivity analasis and plots npv
def plot_variables_npv_total_cost(total_costs):
    npv_values = []
    for total_cost in total_costs:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_npv = npv(annual_profit, y, total_cost=total_cost, lifetime=lifetime)
        npv_values.append(my_npv)
    print("npv for total energy", npv_values)
    plot = plt.figure()
    plt.plot(total_costs, npv_values)
    plt.ylabel('npv')
    plt.xlabel('total cost')
    plt.grid()
    plt.show()


def plot_variables_npv_OM(OMs):
    npv_values = []
    for OM in OMs:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_npv = npv(annual_profit, y, total_cost=total_cost, lifetime=lifetime)
        npv_values.append(my_npv)
    print("npv for OM", npv_values)
    plot = plt.figure()
    plt.plot(OMs, npv_values)
    plt.ylabel('npv')
    plt.xlabel('OM')
    plt.grid()
    plt.show()


def plot_variables_npv_discount_rate(discount_rates):
    npv_values = []
    for discount_rate in discount_rates:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_npv = npv(annual_profit, y, total_cost=total_cost, lifetime=lifetime)
        npv_values.append(my_npv)
    print("npv for discount rates", npv_values)
    plot = plt.figure()
    plt.plot(discount_rates, npv_values)
    plt.ylabel('npv')
    plt.xlabel('discount rate')
    plt.grid()
    plt.show()


def plot_variables_npv_total_energies(total_energies):
    npv_values = []
    for total_energy in total_energies:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_npv = npv(annual_profit, y, total_cost=total_cost, lifetime=lifetime)
        npv_values.append(my_npv)
    print("npv for total energy", npv_values)
    plot = plt.figure()
    plt.plot(total_energies, npv_values)
    plt.ylabel('npv')
    plt.xlabel('total energy')
    plt.grid()
    plt.show()


def plot_variables_npv_lifetime(lifetimes):
    npv_values = []
    for lifetime in lifetimes:
        lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                          discount_rate=discount_rate, total_energy=total_energy)
        profits = market_price - lcoe_value
        annual_profit = (profits * total_energy) - OM
        y = 1 + discount_rate
        my_npv = npv(annual_profit, y, total_cost=total_cost, lifetime=lifetime)
        npv_values.append(my_npv)
    print("npv for lifetimes", npv_values)
    plot = plt.figure()
    plt.plot(lifetimes, npv_values)
    plt.ylabel('npv')
    plt.xlabel('lifetime')
    plt.grid()
    plt.show()


# % % SKRIPT

OM = 100000  # yearly operation and maintenance costs in €
total_cost = 2000000  # capital costs in €
discount_rate = 0.02  # applied discount rate
lifetime = 25  # lifetime of the project in years
total_energy = 3709.552195  # yearly energy produced by the power plant in mwh
market_price = 81.27  # average day ahead market price

total_costs = [1000000, 1500000, 2000000, 2500000]  # capital costs in €
OMs = [10000, 20000, 30000, 40000]  # yearly operation and maintenance costs in €
lifetimes = [25, 30, 35, 40]  # lifetime of the project in years
discount_rates = [0.02, 0.04, 0.08, 0.16]  # applied discount rate
total_energies = [3000, 3709.552195, 4000, 4500]  # yearly energy produced by the power plant in mwh

# plots
plot_variable_total_cost(total_costs)
plot_variable_OM(OMs)
plot_variable_discount_rate(discount_rates)
plot_variable_lifetimes(lifetimes)
plot_variable_total_energies(total_energies)
plot_variable_irr_total_cost(total_costs)
plot_variable_irr_OM(OMs)
plot_variable_irr_discount_rates(discount_rates)
plot_variable_irr_lifetimes(lifetimes)
plot_variable_irr_total_energy(total_energies)
plot_variables_npv_total_cost(total_costs)
plot_variables_npv_OM(OMs)
plot_variables_npv_discount_rate(discount_rates)
plot_variables_npv_total_energies(total_energies)
plot_variables_npv_lifetime(lifetimes)


def test():
    OMs = [100000, 20000, 300000, 40000]
    discount_rates = [0.02, 0.03, 0.04, 0.05]
    lifetimes = [25, 30, 35, 40]
    total_energy = 3709.552195
    market_price = 81.27
    list_costs = []

    for total_cost in total_costs:
        for OM in OMs:
            for discount_rate in discount_rates:
                for lifetime in lifetimes:

                    lcoe_value = lcoe(total_cost=total_cost, OM=OM, lifetime=lifetime,
                                      discount_rate=discount_rate)  # calculation lcoe
                    # calculation profit per mwh
                    profits = market_price - lcoe_value
                    if lcoe_value < market_price:

                        print("profits per mWh:", profits)
                    else:
                        print("lcoe is above market price")
                    annual_profit = (profits * total_energy) - OM
                    y = 1 + discount_rate

                    my_irr = irr(lifetime=lifetime, total_cost=total_cost)
                    my_npv = npv(annual_profit, y, total_cost=total_cost, lifetime=lifetime)

                    # print(my_irr)
                    # print(my_npv)
                    print("annual revenue is:", annual_profit)
                    print(f"irr is {my_irr} and npv is {my_npv}")
