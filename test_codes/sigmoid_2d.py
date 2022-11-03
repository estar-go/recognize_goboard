import torch




if __name__ == '__main__':
    # t1 = torch.randn(4)
    # print(t1)
    # st1 = torch.sigmoid(t1)
    # print(st1, st1.sum())
    batch_size = 2
    t2 = torch.randn(batch_size, 3, 5, 5)
    # print(t2)
    st2 = torch.special.expit(t2)
    mst2 = torch.argmax(st2, dim=1)
    # st2 = torch.softmax(t2, dim=0)
    # print(st2)
    # print(st2[0, :, 0, 0].sum())

    gt2 = torch.special.expit(torch.randn(batch_size, 3, 5, 5))
    # gt2 = torch.softmax(torch.randn(3, 5, 5), dim=0)
    print(gt2)
    mgt2 = torch.argmax(gt2, dim=1)

    # print(mgt2.shape)
    print(mst2)
    print(mgt2)

    # print('convert label to onehot')
    # onehot_gt2 = torch.zeros((3, 5, 5))
    # onehot_gt2[0, mgt2==0] = 1
    # onehot_gt2[1, mgt2==1] = 1
    # onehot_gt2[2, mgt2==2] = 1
    # print(onehot_gt2)

    loss_func = torch.nn.BCELoss()
    loss = loss_func(st2, gt2)
    print(loss)

    tmp_mst = torch.zeros_like(mst2)
    tmp_mst[0] = mst2[0]
    tmp_mst[1] = mgt2[1]
    print(tmp_mst)
    right = torch.sum(torch.tensor([torch.equal(tmp_mst[i], mgt2[i]) for i in range(batch_size)])).mul(100.0 / mst2.shape[0]) #.mul_(100.0 / batch_size)
    print(right)
    # for i in range(batch_size):
    #     print(int(torch.equal(tmp_mst[i], mgt2[i])))
    
