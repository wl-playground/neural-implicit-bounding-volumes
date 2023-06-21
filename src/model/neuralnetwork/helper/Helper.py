def hut_3d_calculator(image_width, image_height):
    if image_width is not image_height:
        raise ValueError("training image width and height must be the same for this calculator")

    scaling = 64.0/float(image_width)

    # -0.2*X - 0.2*Y + 13 calculated for 64x64 input
    f = "f = max(0, -{}*X - {}*Y + 13)".format(0.2*scaling, 0.2*scaling)

    # 0.2*X + 0.2*Y - 13 calculated for 64x64 input
    g = "g = max(0, {}*X + {}*Y - 13)".format(0.2*scaling, 0.2*scaling)

    # -0.2*X + 0.2*Y + 5 calculated for 64x64 input, make 5 bigger for a bigger hut, and smaller for a smaller hut
    h = "h = max(0, -{}*X + {}*Y + 5)".format(0.2*scaling, 0.2*scaling)

    # -0.4*X + 0.4*Y + 0 calculated for 64x64 input
    j = "j = max(0, -{}*X + {}*Y)".format(0.4*scaling, 0.4*scaling)

    # 3d hut
    k = "hut_3d = max(0, h - j - g - f)"

    print("for a training image of dimensions ({}, {}):\n".format(image_width, image_height))
    print(f)
    print(g)
    print(h)
    print(j, "\n")
    print(k)