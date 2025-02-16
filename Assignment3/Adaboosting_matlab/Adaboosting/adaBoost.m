%%
% File Name: AdaBoost
% This is the implementation of the ada boost algorithm.
% Parameters - very easy to gues by name...
% Return values: i - hypothesis-index  vector.
%                t - threshhols vector
%                beta - weighted beta.
%%
function boosted=adaBoost(train,train_label,cycles)
    disp('running adaBoost algorithm');
    d=size(train);
	distribution=ones(1,d(1))/d(1);  %每一個訓練樣本最開始時都被賦予相同的權值：1/N。
	error=zeros(1,cycles);
	beta=zeros(1,cycles);
	label=(train_label(:)>=5);% contain the correct label per vector

	for j=1:cycles
        if(mod(j,10)==0)
            disp([j,cycles]);
        end
	[i,t]=weakLearner(distribution,train,label);
    error(j)=distribution*abs(label-(train(:,i)>=t)); % 是在第j次迭代中弱分類器的錯誤率
    beta(j)=error(j)/(1-error(j));  %beta(j)表示第j次迭代中的弱分類器的權重 和error成反比 代表錯誤率越小權重越大
    boosted(j,:)=[beta(j),i,t];
   

    distribution=distribution.* exp(log(beta(j))*(1-abs(label-(train(:,i)>=t))))';
% 計算當前迭代得到的弱分類器的加權預測結果，用於更新每個樣本的權重distribution。
% 對於每個樣本，如果它被弱分類器正確分類，那麼它的權重就會下降；如果它被錯誤分類，那麼它的權重就會上升。
% 這樣，被分錯的樣本在下一次迭代中就會更有可能被選中，從而讓後面的弱分類器更關注這些被分錯的樣本。

    distribution=distribution/sum(distribution);   %Normalization每個資料點的權重以保證它們總和為1
    
    end


for k = 1:3
    disp(['Blending weight of weak learner ' num2str(k) ': ' num2str(beta(k))]);
end

    