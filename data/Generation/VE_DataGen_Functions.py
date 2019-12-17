#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy as sci
import scipy.integrate as integ


def Eval_Array_From_Expression(Input_Value_Array, Symbol_Variable, Symbolic_Expression):
    
    try:
        Eval_Func = sp.lambdify(Symbol_Variable, Symbolic_Expression)
        Result_Array = Eval_Func(Input_Value_Array)
        if len(Result_Array) != len(Input_Value_Array):
            raise
    except:
        Result_Array = np.array([])
        for Input_Value in Input_Value_Array:
            Result_Array = np.append(Result_Array, Symbolic_Expression.evalf(subs={Symbol_Variable: Input_Value}))
    
    return Result_Array


def Plot_Strain_Stress(title, time, Strain_Eval, Stress_Eval):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)')

    #Set up Strain on first axis
    colour = 'tab:blue'
    ax1.set_ylabel('Strain', color=colour)
    ax1.plot(time, Strain_Eval, color=colour) # This is the key step that actually puts the data on the axes. The rest is asthetic.
    ax1.tick_params(axis='y', labelcolor=colour)

    #Create Second axis. twinx means that the new axes share the same x axis. twiny also exists.
    ax2 = ax1.twinx()

    #Set up Stress on second axis
    colour = 'tab:red'
    ax2.set_ylabel('Stress (N m^-2)', color=colour)  # we already handled the x-label with ax1
    ax2.plot(time, Stress_Eval, color=colour)
    ax2.tick_params(axis='y', labelcolor=colour)


    align_yaxis(ax1, ax2) # Function not of my design.
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.title(title)

    plt.show()


# This is a function I have stolen from Pietro Battiston's answer at
# https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])


def Maxwell_Sum(Elastic_Moduli, Viscosities):
    
    s = sp.symbols('s')
    
    Total_Laplace_Sum = sp.S(0)
    for Branch_Index in range(len(Elastic_Moduli)):
        Branch_Laplace_Sum = 1/(sp.S(1)/Elastic_Moduli[Branch_Index] + 1/(s*Viscosities[Branch_Index]))
        Total_Laplace_Sum += Branch_Laplace_Sum
    
    return Total_Laplace_Sum


def Kelvin_Sum(Elastic_Moduli, Viscosities):
    
    s = sp.symbols('s')
    
    Total_Laplace_Sum = sp.S(0)
    for Element_Index in range(len(Elastic_Moduli)):
        Element_Laplace_Sum = 1/(Elastic_Moduli[Element_Index] + s*Viscosities[Element_Index])
        Total_Laplace_Sum += Element_Laplace_Sum
    
    return Total_Laplace_Sum


def Stress_Strain_Master(Input_Type, Model_Type, E_Mods, Viscs, Input_Function):
    
    if len(E_Mods) - len(Viscs) != 1:
        print('Model parameters do not match. There should be one greater number of Elastic Moduli to the number of Viscosities.')
        return None
    
    t = sp.symbols('t', real=True, positive=True)
    s = sp.symbols('s')
    
    L_Input, Min_s, Aux_Cond = sp.laplace_transform(Input_Function, t, s)
    
    Post_Inverse_Addition = 0
    
    if Input_Type == 'Strain' and Model_Type == 'GKM':
        To_Inverse = (1/(sp.S(1)/E_Mods[0] + Kelvin_Sum(E_Mods[1:], Viscs)))*L_Input
    elif Input_Type == 'Stress' and Model_Type == 'GKM':
        To_Inverse = Kelvin_Sum(E_Mods[1:], Viscs)*L_Input
        Post_Inverse_Addition = Input_Function/E_Mods[0]
    elif Input_Type == 'Stress' and Model_Type == 'GMM':
        To_Inverse = (sp.S(1)/(E_Mods[0] + Maxwell_Sum(E_Mods[1:], Viscs)))*L_Input
    elif Input_Type == 'Strain' and Model_Type == 'GMM':
        To_Inverse = Maxwell_Sum(E_Mods[1:], Viscs)*L_Input
        Post_Inverse_Addition = Input_Function*E_Mods[0]
    else:
        print('Improper description of input or model choice')
        return None
    
    Inversed_Expression = sp.inverse_laplace_transform(To_Inverse, s, t)
    
    Response_Expression = Inversed_Expression + Post_Inverse_Addition
    
    return Response_Expression


def Eval_Graph_Strain_Stress(title, Time_Points, Input_Function, Tuple_of_Expressions, Input_Type, Int_Type, t):
    
    if Input_Type == 'Strain':
        Strain_Eval = Eval_Array_From_Expression(Time_Points, t, Input_Function)
    elif Input_Type == 'Stress':
        Stress_Eval = Eval_Array_From_Expression(Time_Points, t, Input_Function)
    else:
        print('Improper description of input choice')
        return
    
    if Int_Type == 'Analytical' and Input_Type == 'Strain':
        Stress_Eval = Eval_Array_From_Expression(Time_Points, t, Tuple_of_Expressions)
    elif Int_Type == 'Analytical' and Input_Type == 'Stress':
        Strain_Eval = Eval_Array_From_Expression(Time_Points, t, Tuple_of_Expressions)
    elif Int_Type == 'Numerical' and Input_Type == 'Strain':
        Stress_Eval = Eval_Array_From_Expr_w_NumInt(Time_Points, t, Tuple_of_Expressions)
    elif Int_Type == 'Numerical' and Input_Type == 'Stress':
        Strain_Eval = Eval_Array_From_Expr_w_NumInt(Time_Points, t, Tuple_of_Expressions)
    else:
        print('Improper description of integration type')
        return
    
    Strain_Eval = Strain_Eval.real
    Stress_Eval = Stress_Eval.real
    
    Plot_Strain_Stress(title, Time_Points, Strain_Eval, Stress_Eval)
    
    return Strain_Eval, Stress_Eval

    
def Stress_Strain_Master_Int(Input_Type, E_Mods, Viscs, Input_Function, Int_Type, t):
    
    if len(E_Mods) - len(Viscs) != 1:
        print('Model parameters do not match. There should be one greater number of Elastic Moduli to the number of Viscosities.')
        return None
    
    x = sp.symbols('x', real=True)
    
    if Input_Type == 'Strain':
        Creep_Relax_Expression = Relax_Sum(E_Mods, Viscs, t)
    elif Input_Type == 'Stress': 
        Creep_Relax_Expression = Creep_Sum(E_Mods, Viscs, t)
    else:
        print('Improper description of input choice')
        return
    
    D_Input = Input_Function.diff(t)
    To_Integrate = Creep_Relax_Expression.subs(t, t-x)*D_Input.subs(t, x)
    
    First_Term = Input_Function.evalf(subs={t: 0})*Creep_Relax_Expression
    
    if Int_Type == 'Analytical':
        Second_Term = sp.integrate(To_Integrate, (x, 0, t))
        Analytical_Term = First_Term + Second_Term
        return Analytical_Term
    elif Int_Type == 'Numerical':
        # Almost undoubtedly, this will not work with a heaviside function until patched.
        # lambdify uses numpy and scipy functions. If integ.quad fails, try just scipy ones.
        Lambdified_To_Int = sp.lambdify([x, t], To_Integrate)#, modules='scipy')
        Numerical_Term_Func = lambda t: integ.quad(Lambdified_To_Int, 0, t, args=(t))
        Analytical_Term = First_Term
        return Analytical_Term, Numerical_Term_Func
    else:
        print('Improper description of integration type')
        return


def Relax_Sum(E_Mods, Viscs, t):
    
    Relax_Sum_Expression = E_Mods[0]
        
    for i in range(len(Viscs)):
        Relax_Sum_Expression += E_Mods[i+1]*sp.exp((-1*t*E_Mods[i+1])/Viscs[i])
        
    return Relax_Sum_Expression


def Creep_Sum(E_Mods, Viscs, t):
    
    Relax_Sum_Expression = sp.S(1)/E_Mods[0]
        
    for i in range(len(Viscs)):
        Relax_Sum_Expression += (sp.S(1)/E_Mods[i+1])*(1-sp.exp((-1*t*E_Mods[i+1])/Viscs[i]))
        
    return Relax_Sum_Expression


def Eval_Array_From_Expr_w_NumInt(Input_Value_Array, Symbol_Variable, Tuple_of_Expressions):
    
    Analytical_Term = Tuple_of_Expressions[0]
    Numerical_Term_Func = Tuple_of_Expressions[1]
    Result_Array = np.array([])
    for Input_Value in Input_Value_Array:
        Ana_Value = Analytical_Term.evalf(subs={Symbol_Variable: Input_Value})
        Num_Value, Num_Error = Numerical_Term_Func(Input_Value)
        Result_Array = np.append(Result_Array, Ana_Value+Num_Value)
    
    return Result_Array