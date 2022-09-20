function cutPoints = cutPointsForSubset(attribute,labels, first, last,alpha)

currentCutPoint = - realmax('double');
bestCutPoint = -1;
bestIndex = -1;
numCutPoints = 0;
l  =0;
if last - first< 2
    cutPoints = [];
else
    num_classes = numel(unique(labels));
    N_0 = numel(attribute)+numel(attribute)/9;
    counts = zeros(2,num_classes);
    num_Inst = numel(attribute(first:last-1));
    [c_label,~,ic] = unique(labels(first:last-1));
    counts(2,c_label) = accumarray(ic,1)';

    priorCounts = counts(2,:);
    priorEntropy =entropy(priorCounts);
    bestEntropy = priorEntropy;

    bestCounts = zeros(2,num_classes);
    for i=first:(last-2)
        counts(1,labels(i)) = counts(1,labels(i)) +1;
        counts(2,labels(i)) = counts(2,labels(i)) -1;

        if attribute(i) < attribute(i+1)
            currentCutPoint = (attribute(i) + attribute(i+1)) /2.0;
            currentEntropy = entropyConditionedOnRows(counts);

            if currentEntropy < bestEntropy 
                bestEntropy = currentEntropy;
                bestCutPoint = currentCutPoint;

                bestIndex = i;
                bestCounts = counts;
            end
            numCutPoints = numCutPoints +1;
        end  
    end
    numCutPoints = last - first -1;
    gain = priorEntropy - bestEntropy;
    if gain <= 0
        cutPoints = [];
    else
        accept = FayyadAndIranisMDL(priorCounts,bestCounts,num_Inst,numCutPoints,alpha,N_0);
        if accept
            l = l+1;
            left = cutPointsForSubset(attribute,labels,first,bestIndex+1,alpha);
            right = cutPointsForSubset(attribute,labels,bestIndex+1,last,alpha);
            if isempty(left)&& isempty(right)
                cutPoints = bestCutPoint;
            elseif isempty(right)
                cutPoints = [left, bestCutPoint];
            elseif isempty(left)
                cutPoints = [bestCutPoint,right];
            else
                cutPoints = [left,bestCutPoint,right];
            end
        else
            cutPoints = [];
        end

    end
end