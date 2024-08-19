
def control_statements():
    pass   #  shits to easy

def collection_ex1():
    a = [1, 3, 5]
    b = [2, 4, 6]
    c = a + b
    print(f"c:{c}")
    d = sorted(c)
    print(f"c:{c}")
    print(f"d:{d}")
    d = d[::-1]
    print(d)
    c[3] = 42
    print(c)
    d.append(10)
    c.extend([7,8,9])
    print(c[:3])
    print(d[-1])
    print(len(d))
