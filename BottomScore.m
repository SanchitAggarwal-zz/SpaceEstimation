function BS = BottomScore(AllLines,PolyX,PolyY)
    [M,N] = size(AllLines);
    BS = zeros(M,1);
    K=1;
    for K=1:M
        D = 0;
        if AllLines(K,1)<AllLines(K,2)
            X1 = AllLines(K,1);
            X2 = AllLines(K,2);
            Y1 = AllLines(K,3);
        else
            X1 = AllLines(K,2);
            X2 = AllLines(K,1);
            Y1 = AllLines(K,4);
        end
        for X = X1:X2
            Y = (X-X1)*tan(AllLines(K,5))+Y1;
            d = abs(p_poly_dist(X, Y, PolyX, PolyY));
            D = D + d;
        end
        D = exp(-D/1800);
        BS(K)=D;
    end
end