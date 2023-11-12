library(fase)
library(readr)
library(abind)
library(pROC)

seed = 10
n_time_points = 10

logit = function(x) {
    log(x/(1-x))
}

dir_name = paste0('data_', seed)

time_points = readr::read_table(paste0(dir_name, '/time_points.npy'), col_names = FALSE)$X1
A = NULL
probas = NULL
for (t in 1:n_time_points) {
    A = abind(A, read.table(paste0(dir_name, '/Y_', t, '.npy')), along = 3)
    probas = rbind(probas, read.table(paste0(dir_name, '/proba_', t, '.npy'))[,1])
}
print(dim(A))

# hollowization for square matrices
hollowize <- function(M){
    M - diag(diag(M))
}

# hollowization for 3D arrays
hollowize3 <- function(A){
    array(apply(A,3,hollowize),dim(A))
}

Z_to_Theta <- function(Z,self_loops = FALSE){
    Theta <- array(apply(Z,3,tcrossprod),
                   c(dim(Z)[1],dim(Z)[1],dim(Z)[3]))
    if(self_loops){
        return(Theta)
    }
    else{
        return(hollowize3(Theta))
    }
}

# set a seed
set.seed(1)

model_select = matrix(0, nrow = 24,  ncol = 4)
fits = list()
idx = 1
start.time <- Sys.time()
for (q in seq(5, 9, by = 2)) {
    for (d in 1:6) {
        fit <- fase(A,d=d,self_loops=FALSE,
                spline_design=list(type='bs',q=q,x_vec=time_points),
                output_options=list(return_coords=TRUE))
        model_select[idx,] = c(idx, q, d, fit$ngcv)
        print(paste0('q = ', q, ' d = ', d, ' ngcv = ', fit$ngcv))
        fits[[idx]] = fit
        idx = idx + 1
    }
}
end.time <- Sys.time()
time_fase = end.time - start.time

best_idx = which.min(model_select[,4])
fit = fits[[best_idx]]

proba = Z_to_Theta(fit$Z)
tri.indices = upper.tri(proba[,,1])

pred = c()
true = c()
y_vec = c()
for (t in 1:10) {
    pred = c(pred, proba[,,t][tri.indices])
    true = c(true, probas[t,])
    y_vec = c(y_vec, A[,,t][tri.indices])
}
ppc_fase = cor(true, pred)
auc_fase = auc(y_vec, pred)

pred = pmin(pmax(pred, 1e-5), 0.999)
logit_fase = sqrt(mean((logit(pred) - logit(true)) ^ 2))

data = data.frame(
    density = 0.1,
    auc_fase = auc_fase,
    ppc_fase = ppc_fase,
    logit_fase = logit_fase,
    q = model_select[best_idx,2],
    d = model_select[best_idx,3],
    time_fase = as.numeric(time_fase, units='secs')
)
