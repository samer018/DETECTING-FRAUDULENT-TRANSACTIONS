rm(list=objects())
install.packages("tidyverse")
install.packages("tibble")
install.packages("parallel")
install.packages("tsoutliers")
install.packages("lmtest")
install.packages("lattice")
install.packages("grid")
install.packages("DMwR")
install.packages("ROI")
install.packages("rpart")
install.packages("tseries")
install.packages("TTR")
install.packages("xts")
install.packages("randomForest")
install.packages("DescTools")
install.packages("rpart.plot")
install.packages("Hmisc")
install.packages("xgboost")
install.packages("Matrix")
install.packages("vcd")
library(graphics)
library(tibble)
library(tidyr)
library(base)
library(dplyr)
library(readr)
library(parallel)
library(lmtest)
library(lattice)
library(grid)
library(DMwR)
library(car)
library(rpart)
library(ROI)
library(class)
library(tseries)
library(xts)
library(quantmod)
library(TTR)
library(randomForest)
library(DescTools)
library(rpart.plot)
library(Hmisc)
library(Matrix)
library(xgboost)
library(data.table)
library(vcd)
setwd("C:/Users/samer/Documents/Master/Apprentissage statistique/TP2")
options("scipen"=100, "digits"=10)
library(DMwR)
data(sales)

#Propriètés statistiques 
summary(sales)
describe(sales)

nlevels(sales$ID)
c(nlevels(sales$ID), nlevels(sales$Prod))

#detection des lignes ou la quantité et la valeur totales sont manquantes les deux? 888
length(which(is.na(sales$Quant) & is.na(sales$Val)))

table(sales$Insp)/nrow(sales) * 100

#Les diagrammes en bâton (a) et (b)
a=table(sales$ID)
tail(a)
barplot(a, main="distribution de ID", 
        xlab="Vendeur",ylab="Nombre de rapports")

b=table(sales$Prod)
tail(b)
barplot(b, main="distribution de produits", 
        xlab="Produit",ylab="Nombre de rapports")

#Ajouter une colonne unit price
sales$Uprice<-sales$Val/sales$Quant
head(sales)

#Boxplot visualisant les variations de cette variable
boxplot(Uprice~Prod,data=sales, main="Les variations des prix unitaires de tous les produits", 
        xlab="Produit", ylab="Prix unitaire")

#Computation of the number of transactions and the average
attach(sales)
num.prod = as.numeric(table(Prod))
sum(num.prod < 20) # nb de produits ayant <20 transactions
av.uprice = tapply(Uprice, Prod, mean, na.rm=T)
av.uprice
tail(av.uprice)

#Compute the boxplot statistics for the distributions of unit prices for each product
BP.uprice=tapply(Uprice, Prod, function(x) boxplot.stats(x)$stats)
length(BP.uprice)
tail(BP.uprice)

#Question 3
n=boxplot.stats(x)$out
x=c(1,1,1,1,1,1,1,1,1,24,25)
n
rm(x)
#1)
out.uprice=tapply(Uprice, Prod, function(x) length(boxplot.stats(x)$out))
length(out.uprice)
tail(out.uprice)

#2)
sum((out.uprice))

#3)
sum((out.uprice))/nrow(sales)*100

#4)
x=as.numeric(out.uprice)
for (i in 1:10)
{
print(which.max(x));
x[(which.max(x))]=0;

}
#Supervised techniques
sales1 <- sales[sales$Insp != "unkn",]
sales1$Insp=factor(sales1$Insp) # permet de supprimer la modalité "unkn"
sales1 <- na.omit(sales1) # supprime les exemples avec valeurs manquantes

#Echantillon d'entrainement et l'échantillon de test
sales1 <- sales1[, c("ID", "Prod", "Uprice", "Insp")]
N <- nrow(sales1)
Index <- sample(1:N)
K=round(0.7*N)
sales.train <- sales1[Index[1:K],]
sales.test <- sales1[Index[(K+1):N],]


#modèle )question 4
train <- data.table(sales.train, keep.rownames = F)
head(train)
test<- data.table(sales.test, keep.rownames = F)
head(test)
sparse_matrix1 <- sparse.model.matrix(Insp~.-1, data = train)
head(sparse_matrix1)
dim(sparse_matrix1)
label1 = train[,Insp] == "ok"

sparse_matrix2 <- sparse.model.matrix(Insp~.-1, data = test)
label2 = test[,Insp] == "ok"

#xgb.DMatrix
dtrain <- xgb.DMatrix(data = sparse_matrix1, label=label1)
dtest <- xgb.DMatrix(data = sparse_matrix2, label=label2)
watchlist <- list(train=dtrain, test=dtest)

bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nthread = 2, nrounds=10, watchlist=watchlist, objective = "binary:logistic")

#prédiction par l'arbre de décision
library(rpart)
train1=as.data.frame(sales.train)
test1=as.data.frame(sales.test)
rt.insp = rpart(Insp ~ ., data = train1,method="class")
rt.prediction.insp=predict(rt.insp,test1,type="class")
t=table(rt.prediction.insp,test1[,"Insp"])
#erreur de prediction
Erreur=(sum(t)-t[1,1]-t[2,2])/sum(t)
Erreur

#The class imbalance 
newData <- SMOTE(Insp ~ ., sales.train, perc.under = 500)
table(newData$Insp)
table(sales.train$Insp)

#De nouveau
train <- data.table(newData, keep.rownames = F)
head(train)
test<- data.table(sales.test, keep.rownames = F)
head(test)
sparse_matrix1 <- sparse.model.matrix(Insp~.-1, data = train)
head(sparse_matrix1)
dim(sparse_matrix1)
label1 = train[,Insp] == "ok"

sparse_matrix2 <- sparse.model.matrix(Insp~.-1, data = test)
label2 = test[,Insp] == "ok"

#xgb.DMatrix
dtrain <- xgb.DMatrix(data = sparse_matrix1, label=label1)
dtest <- xgb.DMatrix(data = sparse_matrix2, label=label2)
watchlist <- list(train=dtrain, test=dtest)

bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nthread = 2, nrounds=10, watchlist=watchlist, objective = "binary:logistic")

#prédiction par l'arbre de décision
library(rpart)
train1=as.data.frame(newData)
test1=as.data.frame(sales.test)
rt.insp = rpart(Insp ~ ., data = train1,method="class")
rt.prediction.insp=predict(rt.insp,test1,type="class")
t=table(rt.prediction.insp,test1[,"Insp"])
#erreur de prediction
Erreur=(sum(t)-t[1,1]-t[2,2])/sum(t)
Erreur

#logistic model
train2=train1
test2=test1
train2[,"Insp"]=train2[,"Insp"]=="ok"
test2[,"Insp"]=test2[,"Insp"]=="ok"
head(train2)
head(test2)
pred<-glm(formula = Insp ~., data=train2, family = binomial(link = logit))
g.pred=predict(pred,test1,type="class")
t=table(g.pred,test1[,"Insp"])
#erreur de prediction
Erreur=(sum(t)-t[1,1]-t[2,2])/sum(t)
Erreur