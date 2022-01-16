import itertools
import sys


def Povm_List_Qubit_1(standard_list = ['d', 'a', 'r', 'l','h', 'v']):

    return standard_list

def Povm_List_Qubit_2():
    standard_list = Povm_List_Qubit_1()
    iter_for = [['d', 'a'], ['r', 'l'], ['h', 'v']]
    povm_list = []
    for j in iter_for:
        povm_list.append(list(itertools.product(standard_list, j)))
    p_list = []
    for i in range(len(povm_list)):
        p_list.append([list(j) for j in povm_list[i]])

    return p_list

def Povm_List_Qubit_3():
    dd,rr,hh = Povm_List_Qubit_2()
    iter_for = [['d', 'a'],['r', 'l'],['h', 'v']]
    povm_list = []
    for j in iter_for:
        for k in [dd,rr,hh]:
            povm_list.append(list(itertools.product(k,j)))
    p_list = []
    for i in range(len(povm_list)):
        p_list.append([list(j) for j in povm_list[i]])


    return p_list


def Povm_List_Qubit_4():
    dd_2, dd_3, dd_4, rr_2, rr_3, rr_4, hh_2, hh_3, hh_4 = Povm_List_Qubit_3()
    iter_for = [['d', 'a'], ['r', 'l'], ['h', 'v']]
    povm_list = []
    for j in iter_for:
        for k in [dd_2, dd_3, dd_4, rr_2, rr_3, rr_4, hh_2, hh_3, hh_4]:
            povm_list.append(list(itertools.product(k,j)))
    p_list = []
    for i in range(len(povm_list)):
        p_list.append([list(j) for j in povm_list[i]])
    return p_list

def Convert_to_Projections(raw_povm_list,qs = 3):
    pp_list = []
    for i in raw_povm_list:
        for k in i:
            un_list = list(itertools.chain.from_iterable(k))
            pp_list.append(un_list)
    if qs > 3:
        ppp_list = []
        for i in pp_list:
            un_list = list(itertools.chain.from_iterable(i))
            ppp_list.append(un_list)
        pp_list = ppp_list
    return pp_list

def Basis_Order(qs=2):
    if qs == 1:
        Plist = Povm_List_Qubit_1()
    elif qs == 2:
        Plist = Povm_List_Qubit_2()
    elif qs == 3:
        Plist = Povm_List_Qubit_3()
    elif qs == 4:
        Plist = Povm_List_Qubit_4()
    else:
        sys.exit('Qubit size must be less or equal to 4.')
    converted = Convert_to_Projections(Plist, qs=4)
    titles = [''.join(i) for i in converted]
    return titles


# titles = Basis(qs=4)
# print(titles)
# print (len(titles))
