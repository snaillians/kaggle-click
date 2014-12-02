setwd("github/kaggle-click")
source("r/utilities.r")

wdata=read.csv("data/train.csv",nrows=100000,colClasses=c("character","numeric","character",rep("factor",21)))
wdata=wdata[,c(2,1,3:24)]
wdata$hour=as.factor(substring(wdata$hour,7,8))
#at=transform(wdata$click,wdata[,4:5])$map
temp=transformfactor(wdata$click,wdata[,c(3:11,14:24)])
wd2=wdata
wd2[,c(3:11,14:27)]=temp$dataframe
wd2[,12:13]=hash(wdata[,12:13])
map=temp$map
head(wd2)
n1=dim(wdata)[1]
sub=sample(1:n1,n1*0.5)
train=wd2[sub,]
validate=wd2[-sub,]

wtest=read.csv("data/test_rev2.csv",nrows=100000,colClasses=c("character","character",rep("factor",24)))
wtest=cbind(click=rep(0,dim(wtest)[1]),wtest)
wtest$hour=as.factor(substring(wtest$hour,7,8))
#tail(transfactor(tt[,9:16],map[7:14]))
wt2=wtest
wt2[,c(3:11,14:27)]=transfactor(wtest[,c(3:11,14:27)],map)
wt2[,12:13]=hash(wtest[,12:13])
head(wt2)

cbind(nlevel=sapply(wdata[3:27],nlevels),Entropy=sapply(wdata[,3:27],Entropy),mutualinfo=sapply(wdata[,3:27],mutateInfo,wdata$click))
#fol=formula(click~.-id)
fol=formula(click~site_id+site_domain+site_category+app_id+app_domain+app_category
            +device_id+device_ip+device_model+device_geo_country+C17+C18+C19+C20+C21+C22+C24)

pt=mean(train$click) 
prior.validate=rep(pt,dim(validate)[1])
evaluate(prior.validate,validate$click)
prior.predict=rep(pt,dim(test)[1])

glm.fit<-glm(fol,data=train,family=binomial,x=FALSE,y=FALSE,model=FALSE)
glm.validate<-predict(glm.fit,validate,type="response")
#glm.validate[is.na(glm.validate)]<-mean(train$click)
evaluate(glm.validate,validate$click)

lda.fit<-lda(fol,data=train)
lda.validate<-predict(lda.fit,validate)$posterior
evaluate(lda.validate,validate$click)

glm.predict=predict(glm.fit,wt2,type=c"response")
result=cbind(wtest$id,glm.predict)
colnames(result)<-c("id","click")
head(result)
write.csv(result,"data/submission.csv",row.names=FALSE,quote=FALSE)

#-------------------------------------------------------------------
rf.fit<-randomForest(fol,data=train)
rf.validate=predict(rf.fit,validate,'prob')
evaluate(rf.validate[,2],validate$click)
rf.predict=predict(rf.fit,test,'prob')

svm.fit<-svm(fol,data=train, method="C-classification",kernel="radial", probability=TRUE)
svm.validate<-predict(svm.fit,validate,probability=TRUE)%>%attr("probabilities")
evaluate(svm.validate[,2],validate$click)

nb.fit<-naiveBayes(fol,data=train)
nb.validate=predict(nb.fit,validate,type="raw")
evaluate(nb.predict[,2],test$click)








