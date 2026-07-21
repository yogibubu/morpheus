def write_basic_section(f, charge=0, spin_multiplicity=1, point_group="C1",
                        T=298.15, P_atm=1.0, watson_reduction="S"):
    """
    Writes the section #BASIC in the xyzin file.
    """

    f.write("\n#BASIC\n")
    f.write(f"CHARGE              {int(charge)}\n")
    f.write(f"SPIN_MULTIPLICITY   {int(spin_multiplicity)}\n")
    f.write(f"POINT_GROUP         {point_group}\n")
    f.write(f"Watson Reduction {watson_reduction}\n")
    f.write(f"T_K =                 {float(T):.2f}\n")
    f.write(f"P_atm =             {float(P_atm):.6f}\n")
