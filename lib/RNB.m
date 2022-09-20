function rest = RNB(feat,label)

    rng('default')
    class_lab = unique(label);
    NUM_CLASSES = numel(class_lab);

    %% hyper-parameter for RNB
    alpha = 0.5;

    all_feat = feat;
    all_label = label;
    %% 10-fold cross-validation
    indices = crossvalind('Kfold',all_label,10,'Classes',class_lab);
    for k=1:1:10
        test_id = find(indices ==k);
        train_id = find(indices ~=k);

        te_feat = all_feat(test_id,:);
        te_label = all_label(test_id,:);

        tr_feat = all_feat(train_id,:);
        tr_label = all_label(train_id,:);

        [NUM_TR_SAMPLE,NUM_ATTRI] = size(tr_feat);
        NUM_TE_SAMPLE = numel(te_label);

        %% Prior probability
        prior_prob =zeros(NUM_CLASSES,1);
        for c = 1:NUM_CLASSES
            prior_prob(c) = (numel(find(tr_label == c))+1/numel(class_lab))/(numel(tr_label)+1);
        end

        %% Obtain cut points by using MDLP discretization
        [rows,cols] = size(tr_feat);
        m_cutPoints = zeros(cols,rows);
        count = zeros(cols,1);
        lambda =1/(1+exp(-numel(label)/2000));
        for j = 1:cols
            attribute = tr_feat(:,j);
            [A,I] = sort(attribute);
            labels = tr_label(I);
            temp = cutPointsForSubset(A,labels,1,rows+1,lambda);
            count(j) = numel(temp);
            m_cutPoints(j,1:count(j))= temp;
        end


        %% Transform numerical feature into discrete one
        [rows,cols] = size(tr_feat);
        new_tr_feat = zeros(rows,cols);
        for i = 1:rows
            for j = 1:cols
                cutPoint = m_cutPoints(j,1:count(j));
                [~,idx] =min(abs(tr_feat(i,j)-cutPoint));
                if numel(cutPoint) ==0
                    new_tr_feat(i,j) = 1;
                else
                    if tr_feat(i,j) <= cutPoint(idx)
                        new_tr_feat(i,j) = idx;
                    else
                        new_tr_feat(i,j) = idx+1;
                    end
                end
            end
        end
        [rows,cols] =size(te_feat);
        new_te_feat = zeros(size(te_feat));
        for i = 1:rows
            for j = 1:cols
                cutPoint = m_cutPoints(j,1:count(j));
                [~,idx] =min(abs(te_feat(i,j)-cutPoint));
                if numel(cutPoint) ==0
                    new_te_feat(i,j) = 1;
                else
                    if te_feat(i,j) <= cutPoint(idx)
                        new_te_feat(i,j) = idx;
                    else
                        new_te_feat(i,j) = idx+1;
                    end
                end
            end
        end
        tr_feat = new_tr_feat;
        te_feat = new_te_feat;

        %% Likelihood probability of train set
        trans_prob = zeros(NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE);
        for c =1:NUM_CLASSES
            prob = zeros(NUM_ATTRI,NUM_TR_SAMPLE);
            for a=1:NUM_TR_SAMPLE
                for j =1:NUM_ATTRI
                    attribute = tr_feat(:,j);
                    [m_count,~,ic] = unique(attribute(tr_label==c));
                    mWeight = accumarray(ic,1);
                    id = find(m_count == tr_feat(a,j));
                    if isempty(mWeight)
                        prob(j,a) = 1/numel(unique(attribute))/(sum(mWeight)+1);
                    else
                        if isempty(id)
                            prob(j,a) = 1/numel(unique(attribute))/(sum(mWeight)+1);
                        else
                            prob(j,a) = (mWeight(id)+1/numel(unique(attribute)))/ (sum(mWeight)+1);
                        end
                    end
                end
            end
            trans_prob(:,(1+(c-1)*NUM_TR_SAMPLE):c*NUM_TR_SAMPLE) = prob;
        end
        log_trans_prob = log(trans_prob);
        log_prior_prob = log(prior_prob);
        y = zeros(NUM_TR_SAMPLE*NUM_CLASSES,1);
        diff = [];
        prior_matrix = reshape(repmat(log_prior_prob,1,NUM_TR_SAMPLE)',1,NUM_TR_SAMPLE*NUM_CLASSES);
        for c = 1:NUM_CLASSES
            diff = [diff;tr_label==c];
        end


        %% Objective function and its gradiant function
        fMSE = @(w) sum(sum(reshape((diff' - ...
            (1-w(end))*exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob))./repmat(sum(reshape(exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)-...
            w(end)*exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob))./...
            repmat(sum(reshape(exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)).^2,NUM_CLASSES,NUM_TR_SAMPLE)));
        gMSE = @(w)  [sum(reshape(sum(reshape(...
            -(diff' - ...
            (1-w(end))*exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob))./repmat(sum(reshape(exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)-...
            w(end)*exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob))./...
            repmat(sum(reshape(exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES))...
            .*(1-w(end)).*exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob))./repmat(sum(reshape(exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES).*(log_trans_prob - repmat(reshape(sum(reshape(...
            exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob))./repmat(sum(reshape(exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)...
            .*log_trans_prob,NUM_ATTRI*NUM_TR_SAMPLE,NUM_CLASSES),2),NUM_ATTRI,NUM_TR_SAMPLE),1,NUM_CLASSES))...
            ,NUM_ATTRI*NUM_TR_SAMPLE,NUM_CLASSES),2),NUM_ATTRI,NUM_TR_SAMPLE),2);
            reshape(sum(reshape(...
            -(diff' - ...
            (1-w(end))*exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob))./repmat(sum(reshape(exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)-...
            w(end)*exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob))./...
            repmat(sum(reshape(exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES))...
            .* w(end).* exp(prior_matrix+sum(reshape(repmat(reshape(w(NUM_ATTRI+1:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob))./...
            repmat(sum(reshape(exp(prior_matrix+sum(reshape(repmat(reshape(w(NUM_ATTRI+1:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)...
            .*(log_trans_prob - log_trans_prob .* exp(prior_matrix+sum(reshape(repmat(reshape(w(NUM_ATTRI+1:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob))./...
            repmat(sum(reshape(exp(prior_matrix+sum(reshape(repmat(reshape(w(NUM_ATTRI+1:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)...
            ),NUM_ATTRI,NUM_TR_SAMPLE,NUM_CLASSES),2),NUM_ATTRI*NUM_CLASSES,1);
            sum((diff' - ...
            (1-w(end))*exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob))./repmat(sum(reshape(exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)-...
            w(end)*exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob))./...
            repmat(sum(reshape(exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)).*...
            (exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob))./repmat(sum(reshape(exp(prior_matrix+sum(w(1:NUM_ATTRI).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)-...
            exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob))./...
            repmat(sum(reshape(exp(prior_matrix+sum(reshape(repmat(reshape(w(1+NUM_ATTRI:end-1),NUM_ATTRI,NUM_CLASSES),NUM_TR_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TR_SAMPLE).*log_trans_prob)),NUM_TR_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)))];

        %% weight initialization
        w = [];
        l = zeros(NUM_ATTRI*(NUM_CLASSES+1)+1,1);
        u = ones(NUM_ATTRI*(NUM_CLASSES+1)+1,1);

        fun = @(w)fminunc_wrapper(w, fMSE, gMSE);
        opts    = struct( 'factr', 0, 'pgtol', 1e-7);



        opts.x0 = [ones(NUM_ATTRI*(NUM_CLASSES+1),1);alpha];
        %% optimization
        [wk, ~, ~] = lbfgsb(fun, l, u, opts );


        %% Likelihood probability of test set
        trans_te_prob = zeros(NUM_ATTRI,NUM_CLASSES*NUM_TE_SAMPLE);
        for c =1:NUM_CLASSES
            prob =zeros(NUM_ATTRI,NUM_TE_SAMPLE);
            for a=1:NUM_TE_SAMPLE
                for j =1:NUM_ATTRI
                    attribute = tr_feat(:,j);
                    [m_count,~,ic] = unique(attribute(tr_label==c));
                    mWeight = accumarray(ic,1);
                    id = find(m_count == te_feat(a,j));
                    if isempty(mWeight)
                        prob(j,a) = 1/numel(unique(attribute))/(sum(mWeight)+1);
                    else
                        if isempty(id)
                            prob(j,a) = 1/numel(unique(attribute))/(sum(mWeight)+1);
                        else
                            prob(j,a) = (mWeight(id)+1/numel(unique(attribute)))/ (sum(mWeight)+1);
                        end
                    end
                end
            end
            trans_te_prob(:,(1+(c-1)*NUM_TE_SAMPLE):c*NUM_TE_SAMPLE) = prob;
        end
        %% Regularized posterior of testing samples
        log_te_prob = log(trans_te_prob);
        prior_te_matrix = reshape(repmat(log_prior_prob,1,NUM_TE_SAMPLE)',1,NUM_TE_SAMPLE*NUM_CLASSES);
        predict_y = (1-wk(end))*exp(prior_te_matrix+sum(wk(1:NUM_ATTRI).*log_te_prob))./repmat(sum(reshape(exp(prior_te_matrix+sum(wk(1:NUM_ATTRI).*log_te_prob)),NUM_TE_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES)...
            +wk(end)*exp(prior_te_matrix+sum(reshape(repmat(reshape(wk((1+NUM_ATTRI):end-1),NUM_ATTRI,NUM_CLASSES),NUM_TE_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TE_SAMPLE).*log_te_prob))./...
            repmat(sum(reshape(exp(prior_te_matrix+sum(reshape(repmat(reshape(wk((1+NUM_ATTRI):end-1),NUM_ATTRI,NUM_CLASSES),NUM_TE_SAMPLE,1),NUM_ATTRI,NUM_CLASSES*NUM_TE_SAMPLE).*log_te_prob)),NUM_TE_SAMPLE,NUM_CLASSES),2)',1,NUM_CLASSES);
        pre_y = reshape(predict_y,NUM_TE_SAMPLE,NUM_CLASSES);
        %% get accuracy
        [~,id] = max(pre_y,[],2);
        rest(k) = numel(find(te_label==id))/NUM_TE_SAMPLE;
    end
    disp(mean(rest));
end