library(fase)
library(readr)
library(abind)
library(pROC)
#library(caret)
library(yardstick)
library(tibble)

set.seed(1)

logit = function(x) {
    log(x/(1-x))
}

## Helper functions taken from the FASE package

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


## Run comparison
dir_name = 'fase_data'

time_points = readr::read_table(paste0(dir_name, '/time_points.npy'), col_names = FALSE)$X1
n_time_points = length(time_points)
print(n_time_points)
A = NULL
for (t in 1:n_time_points) {
    A = abind(A, read.table(paste0(dir_name, '/Y_', t, '.npy')), along = 3)
}


idx = 1
model_select = matrix(0, nrow = 18,  ncol = 4)
fits = list()
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
y_vec = c()
for (t in 1:n_time_points) {
    pred = c(pred, proba[,,t][tri.indices])
    y_vec = c(y_vec, A[,,t][tri.indices])
}

#pred = pmin(pmax(pred, 1e-5), 0.999)
auc_fase = auc(y_vec, pred)
f1_fase = f_meas_vec(as.factor(y_vec), as.factor(ifelse(pred > 0.5, 1, 0)), event_level = "second")


data = data.frame(
    auc_fase = auc_fase,
    f1_fase = f1_fase,
    q = model_select[best_idx,2],
    d = model_select[best_idx,3],
    time_fase = as.numeric(time_fase, units='secs')
)


out_file = paste0('result.csv')
write_csv(data, paste0(out_file))
