%read the csv file "PATHAK NEETISH"

data = csvread('PATHAK NEETISH.csv');

%extract training and testing data
data_train =  data(1:1600);
data_test = data(1601:2000);

%there are two ways to estimate the transmission and emission matrices :
%hmmestimate and hmmtrain 
%hmmestimate requires the sequence of states to be present which is not the case here
%so we use hmmtrain with an initial guesses for tranisition and emission
%matrices
%since there  are three states, let us consider an equi-probable
%tranisition matrix
trans_guess = [1./3,1./3,1./3;1./3,1./3,1./3;1./3,1./3,1./3];

%since there are five observations , we guess an equi-probable
%emission matrix
emission_guess = [1./5,1./5,1./5,1./5,1./5;1./5,1./5,1./5,1./5,1./5;1./5,1./5,1./5,1./5,1./5];

%also define  sarting probability for initial state
init_prob = [1./3 1./3 1./3];
init_prob2 = [1 0 0];
%estimate trans and emission matrix will be
[TRANS_EST, EMIS_EST]= hmmtrain(data_train,trans_guess,emission_guess,'Maxiterations',1000,'Tolerance',1e-5);

disp('Transition matrix is: ')
disp(TRANS_EST);

disp('Emission matrix is: ')
disp(EMIS_EST);

%state of the system after 1600 observations
disp('State after 1600 observations');
%state after 1600 observation
p = init_prob * ((TRANS_EST)^1600);
disp(p);

%create an augmented matrix based on the state of the system afetr 1600
%traininig examples
TRANS_HAT = [0 p; zeros(size(TRANS_EST,1),1) TRANS_EST];

EMIS_HAT = [zeros(1,size(EMIS_EST,2)); EMIS_EST];

%generated Data
[seqData,statesData] = hmmgenerate(400,TRANS_HAT,EMIS_HAT);

%generated observations
disp('Generated sequence of Observations');
disp(seqData)

%calculate errors
%calculate SSEdiff will be the errors
diff = (transpose(seqData)-(data_test));
SSE = sum(diff.^2);
disp('SSE');
disp(SSE);

RMSE = rms(diff);
disp('RMSE');
disp(RMSE);

disp('R_square');
meanDataTest = mean(data_test);
meanDataTestVec(1:400) = meanDataTest;
%disp(meanDataTestVec);
SST = sum((transpose(data_test)-meanDataTestVec).^2);
SSR = sum((seqData-meanDataTestVec).^2);
R_sq = SSR/SST;
disp(R_sq);
