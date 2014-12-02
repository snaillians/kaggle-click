library(magrittr)
library(e1071)
library(randomForest)
library(glmnet)
library(entropy)
library(data.table)
library(digest)
library(MASS)

evaluate<-function( prediction, actual) {
  epsilon <- .000000000000001
  yhat <- pmin(pmax(prediction, epsilon), 1-epsilon)
  logloss <- -mean(actual*log(yhat)
                   + (1-actual)*log(1 - yhat))
  return(logloss)
}

Entropy<-function(obs){
  m=nlevels(obs)
  prob=prop.table(table(obs))
  prob=prob[prob>1e-10]
  entropy=-sum(prob*log(prob))
  return (entropy)
}

complevels<-function(obs){
  if (class(obs)!="factor") return(obs) 
  else 
    {eps=0.01
  m=nlevels(obs)
  prob=prop.table(table(obs))
  le=levels(obs)
  levels(obs)[prob<min(1/m,eps)]<-"other"
  return(obs)
  }
}

adaptlevels<-function(obs,base){
  le=levels(obs)
  le2=levels(base)
  levels(obs)[!(le%in%le2)]<-"other"
  return(obs)
}

mutualInfo<-function(obs,ref){
  mi=mi.empirical(table(ref,obs))/sqrt(entropy.empirical(table(ref)))/sqrt(entropy.empirical(table(obs)))
  return(mi)
}

transformfactor<-function(response,dataframe){
  transform1<-function(response,col){
    fq=t(table(response,col))/colSums(table(response,col))
    order=sort(fq[,2],index.return=TRUE)
    m=nlevels(col)
    map=1:m
    names(map)<-levels(col)[order$ix]
    levels(col)[order$ix]<-(1:m)
    prob=prop.table(table(col))
    map[["other"]]=sum(prob*as.numeric(levels(col)))
    col<-as.numeric(levels(col))[col]
    return (list(dataframe=col,map=map))
  }
  
  m=dim(dataframe)[2]
  map=list()
  if (is.null(m))  return(transform1(response,dataframe))
  else{
    for(i in 1:m){
        result=transform1(response,dataframe[,i])
        dataframe[,i]=result$dataframe
        map=c(map,list(result$map))
    }
    return(list(dataframe=dataframe,map=map))
  }
  
  
}

transfactor<-function(dataframe,map){
  transsingle<-function(col,smap){
    m=nlevels(col)
    ave=smap[["other"]]
    newlevel=rep(ave,m)
    oldlevel=levels(col)
    for(i in 1:m){
        if( oldlevel[i] %in% names(smap))
            newlevel[i]=smap[[oldlevel[i]]]
    }
    levels(col)<-newlevel
    col=as.numeric(levels(col))[col]
    return(col)
  }
  if(is.null(ncol(dataframe))){
      dataframe=transsingle(dataframe,map)
  }
  else{
    m=dim(dataframe)[2]
    for(i in 1:m){
      dataframe[,i]=transsingle(dataframe[,i],map[[i]])
  }
  }
  return(dataframe)
}

prior<-function(p,pred=0.088660000001){

  return(-(p*log(pred)+(1-p)*log(1-pred)))
}

hash<-function(str){
  strarray=data.matrix(str)
  m=dim(strarray)[2]
  if(m==1) str<-apply(strarray,1,strtoi,35)
  else{
    for(i in 1:m){
      str[,i]<-apply(as.array(strarray[,i]),1,strtoi,35)
    }
  }
  return(str)
}