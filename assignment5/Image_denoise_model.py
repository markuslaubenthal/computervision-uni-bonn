import cv2
import numpy as np
import maxflow

def compute_pairwise_cost3(I,pairwise_cost_same=0.01, pairwise_cost_diff=0.8,axis=1):
    weight = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if axis == 0:
                if i == (I.shape[0]-1):
                    weight[i,j] = 0
                elif I[i,j] == 0.0 and I[i+1,j] == 0.0:
                    weight[i,j] = pairwise_cost_same
                elif I[i,j] == 1.0 and I[i+1,j] == 1.0:
                    weight[i, j] = pairwise_cost_same
                else:
                    weight[i, j] = pairwise_cost_diff
            else:
                if j == (I.shape[1]-1):
                    weight[i,j] = 0
                elif I[i,j] == 0.0 and I[i,j+1] == 0.0:
                    weight[i,j] = pairwise_cost_same
                elif I[i,j] == 1.0 and I[i,j+1] == 1.0:
                    weight[i, j] = pairwise_cost_same
                else:
                    weight[i, j] = pairwise_cost_diff
    return weight


# def compute_pairwise_cost4(I,label_type = 0, pairwise_cost_same=0, pairwise_cost_diff=1,axis=1):
#     weight = np.zeros_like(I)
#     for i in range(I.shape[0]):
#         for j in range(I.shape[1]):
#             if axis == 0:
#                 if i == (I.shape[0]-1):
#                     weight[i,j] = 0
#                 elif I[i,j] == label_type and I[i+1,j] == label_type:
#                     weight[i,j] = pairwise_cost_same
#                 else:
#                     weight[i, j] = pairwise_cost_diff
#             else:
#                 if j == (I.shape[1]-1):
#                     weight[i,j] = 0
#                 elif I[i,j] == label_type and I[i,j+1] == label_type:
#                     weight[i,j] = pairwise_cost_same
#                 else:
#                     weight[i, j] = pairwise_cost_diff
#     return weight

def compute_pairwise_cost4(I,label_type = 0, pairwise_cost_same=0, pairwise_cost_diff=1,axis=1, direction=-1):
    weight = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if axis == 0:

                ## Boundary pixel
                if direction == 1 and i == (I.shape[0]-1):
                    weight[i, j] = 0
                ## Boundary pixel
                elif direction == -1 and i == 0:
                    weight[i,j] = 0

                ## Edges from \alpha to \alpha (\alpha expansion)
                elif I[i,j] == label_type and I[i + direction,j] == label_type:
                    weight[i,j] = 0

                ## Edges from (\gamma or \beta) to \alpha (\alpha expansion)
                elif I[i, j] != label_type and I[i + direction,j] == label_type:
                    weight[i, j] = pairwise_cost_diff
                ## Edges from (\gamma to \gamma) to (\beta to \beta) (\alpha expansion)
                elif I[i, j] != label_type and I[i + direction,j] != label_type and I[i, j] == I[i + direction,j]:
                    weight[i, j] = pairwise_cost_diff
                ## Edges from (\gamma to \beta) to (\beta to \gamma) (\alpha expansion)
                elif I[i, j] != label_type and I[i + direction,j] != label_type and I[i, j] != I[i + direction,j]:
                    weight[i, j] = 0
                ## Edges from \alpha to (\gamma or \beta) (\alpha expansion)
                elif I[i, j] == label_type and I[i + direction, j] != label_type and I[i, j] != I[i + direction, j]:
                    weight[i, j] = 0
                else:
                    weight[i, j] = -np.log(0.000001)
            else:
                if direction == 1 and j == (I.shape[1]-1):
                    weight[i,j] = 0
                elif direction == -1 and j ==0:
                    weight[i,j] = 0

                ## Edges from \alpha to \alpha (\alpha expansion)
                elif I[i,j] == label_type and I[i,j+direction] == label_type:
                    weight[i,j] = 0
                ## Edges from (\gamma or \beta) to \alpha (\alpha expansion)
                elif I[i, j] != label_type and I[i, j + direction] == label_type:
                    weight[i, j] = pairwise_cost_diff
                ## Edges from (\gamma to \gamma) to (\beta to \beta) (\alpha expansion)
                elif I[i, j] != label_type and I[i, j + direction] != label_type and I[i, j] == I[i, j + direction]:
                    weight[i, j] = pairwise_cost_diff
                ## Edges from (\gamma to \beta) to (\beta to \gamma) (\alpha expansion)
                elif I[i, j] != label_type and I[i, j + direction] != label_type and I[i, j] != I[i, j + direction]:
                    weight[i, j] = 0
                    ## Edges from \alpha to (\gamma or \beta) (\alpha expansion)
                elif I[i, j] == label_type and I[i, j + direction] != label_type and I[i, j] != I[i, j + direction]:
                    weight[i, j] = 0
                else:
                    weight[i, j] = -np.log(0.000001)
    return weight

def compute_unary_cost3(I,label_type=0,rho=0.6):

    # type = 0 is Source
    # type = 1 is Sink
    potential = np.zeros_like(I)+0.0001
    if label_type == 0:
        potential[I == 0] = rho
        potential[I == 1] = 1-rho
    elif label_type == 1:
        potential[I == 0] = 1-rho
        potential[I == 1] = rho

    return -np.log(potential)

def compute_unary_cost4(I,label_type=0,rho=0.8, type=0):

    # type = 0 is Source
    # type = 1 is Sink


    if type == 0:
        potential = np.ones_like(I)*rho
    else:
        potential = np.zeros_like(I)
        if label_type == 0:
            potential[I == 0] = 0
            potential[I == 1] = (1 - rho) / 2
            potential[I == 2] = (1 - rho) / 2
        elif label_type == 1:
            potential[I == 0] = (1 - rho) / 2
            potential[I == 1] = 0
            potential[I == 2] = (1 - rho) / 2
        elif label_type == 2:
            potential[I == 0] = (1 - rho) / 2
            potential[I == 1] = (1 - rho) / 2
            potential[I == 2] = 0
    # potential = np.zeros_like(I)
    # if label_type == 0:
    #     potential[I == 0] = rho
    #     potential[I == 1] = (1 - rho) / 2
    #     potential[I == 2] = (1 - rho) / 2
    # elif label_type == 1:
    #     potential[I == 0] = (1 - rho) / 2
    #     potential[I == 1] = rho
    #     potential[I == 2] = (1 - rho) / 2
    # elif label_type == 2:l
    #     potential[I == 0] = (1 - rho) / 2
    #     potential[I == 1] = (1 - rho) / 2
    #     potential[I == 2] = rho

    return potential

def question_3(I,rho=0.6,pairwise_cost_same=0.01,pairwise_cost_diff=0.5):

    I = I.astype('float')/255.0
    ### Create Graph
    g = maxflow.Graph[float]()

    ### Add pixels as nodes
    nodeids = g.add_grid_nodes(I.shape)

    ### Compute Unaray cost
    U_0 = compute_unary_cost3(I, label_type=0, rho=rho)
    U_1 = compute_unary_cost3(I, label_type=1, rho=rho)

    ### Add terminal edges
    g.add_grid_tedges(nodeids, U_0, U_1)

    ### Add Node edges
    ### Vertical Edges
    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])
    weights = compute_pairwise_cost3(I, pairwise_cost_same=pairwise_cost_same,
                                    pairwise_cost_diff=pairwise_cost_diff, axis=0)
    g.add_grid_edges(nodeids,weights=weights,structure=structure,symmetric=True)

    ### Horizontal edges
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])
    weights = compute_pairwise_cost3(I, pairwise_cost_same=pairwise_cost_same,
                                    pairwise_cost_diff=pairwise_cost_diff, axis=1)
    g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=True)
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)



    ### Maxflow
    g.maxflow()

    sgm = g.get_grid_segments(nodeids)
    img2 = np.int_(np.logical_not(sgm))

    cv2.imshow('Original Img', I.astype('uint8') * 255),\
    cv2.imshow('Denoised Img', img2.astype('uint8')*255), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I,rho=0.6):
    I_orig = I.copy()
    I = np.round(I.astype('float') / 128.0)

    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    for label in labels:

        ### Create Graph
        g = maxflow.Graph[float]()
        ### Add pixels as nodes
        nodeids = g.add_grid_nodes(I.shape)

        ### Compute Unaray cost
        U_0 = compute_unary_cost4(I, label_type=label, rho=rho, type=0)
        U_1 = compute_unary_cost4(I, label_type=label, rho=rho, type=1)
        U_0 = -np.log(U_0 + 0.000001)
        U_1 = -np.log(U_1 + 0.000001)

        ### Potts Model
        ### Same node = 0
        ### Different label node = 1

        ### Add terminal edges
        g.add_grid_tedges(nodeids, U_0, U_1)

        ### Add Node edges
        ### Vertical Edges
        structure = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 1, 0]])
        weights = compute_pairwise_cost4(I, label_type=label, pairwise_cost_same=0,
                                         pairwise_cost_diff=1, axis=0, direction=1)
        g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=False)

        structure = np.array([[0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0]])
        weights = compute_pairwise_cost4(I, label_type=label, pairwise_cost_same=0,
                                         pairwise_cost_diff=1, axis=0, direction=-1)
        g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=False)

        ### Horizontal edges
        structure = np.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 0, 0]])
        weights = compute_pairwise_cost4(I, label_type=label, pairwise_cost_same=0,
                                         pairwise_cost_diff=1, axis=1,direction=1)
        g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=False)

        structure = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]])
        weights = compute_pairwise_cost4(I, label_type=label, pairwise_cost_same=0,
                                         pairwise_cost_diff=1, axis=1,direction=-1)
        g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=False)
        # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

        ### Maxflow
        g.maxflow()

        sgm = g.get_grid_segments(nodeids)
        Denoised_I[sgm] = label

    Denoised_I[Denoised_I == 1] = 128
    Denoised_I[Denoised_I == 2] = 255
    Denoised_I = Denoised_I.astype('uint8')
    cv2.imshow('Original Img', I_orig), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.15)
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.3)
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.6)

    ### Call solution for question 4
    #question_4(image_q4, rho=0.1)
    return

if __name__ == "__main__":
    main()
