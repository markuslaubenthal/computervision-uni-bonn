import cv2
import numpy as np
import maxflow

def question_3(I,rho=0.7,pairwise_cost_same=0.005,pairwise_cost_diff=0.2):

    ### 1) Define Graph
    size = I.shape[0] * I.shape[1]
    g = maxflow.Graph[float](size, size * 4)

    ### 2) Add pixels as nodes
    nodes = g.add_nodes(size)

    def index(x,y):
        if x >= I.shape[1] or x < 0 or y >= I.shape[0] or y < 0:
            return None
        return (I.shape[1]*y+x, (x,y))

    def pairwise_cost(p1,p2):
        return pairwise_cost_same if (I[p1[1],p1[0]] == I[p2[1],p2[0]]) else pairwise_cost_diff

    def unary_cost(p1):
        return rho if (I[p1[1],p1[0]] == 255) else 1-rho


    ### 3) Compute Unary cost
    ### 4) Add terminal edges
    ### 5) Add Node edges

    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            # get node indices for current node / left / right / top / bot
            i, il, ir, it, ib = index(x,y), index(x-1,y), index(x+1,y), index(x,y-1), index(x,y+1)

            ### Horizontal edges
            if il is not None:
                g.add_edge(nodes[i[0]],nodes[il[0]], pairwise_cost(i[1],il[1]), pairwise_cost(i[1],il[1]))
            if ir is not None:
                g.add_edge(nodes[i[0]], nodes[ir[0]], pairwise_cost(i[1],ir[1]), pairwise_cost(i[1],ir[1]))
            ### Vertical Edges
            if it is not None:
                g.add_edge(nodes[i[0]],nodes[it[0]], pairwise_cost(i[1],it[1]), pairwise_cost(i[1],it[1]))
            if ib is not None:
                g.add_edge(nodes[i[0]],nodes[ib[0]], pairwise_cost(i[1],ib[1]), pairwise_cost(i[1],ib[1]))

            g.add_tedge(nodes[i[0]], 1-unary_cost(i[1]), unary_cost(i[1]))




    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)



    ### 6) Maxflow
    g.maxflow()

    # Graph cut denoise image
    Denoised_I = I.copy().astype(np.float32)
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            Denoised_I[y,x] = g.get_segment(nodes[index(x,y)[0]])


    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()
    return None

def question_4(I,rho=0.6):

    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    ### Use Alpha expansion binary image for each label

    ### 1) Define Graph

    ### 2) Add pixels as nodes

    ### 3) Compute Unary cost

    ### 4) Add terminal edges

    ### 5) Add Node edges
    ### Vertical Edges

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow


    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.35)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.55)

    ### Call solution for question 4
    #question_4(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()
