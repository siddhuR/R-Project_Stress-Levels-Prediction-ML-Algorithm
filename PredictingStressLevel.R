# Predicting Stress Levels Using Machine Learning Algorithms

#dataset of stresslevel.csv
data=read.csv("StressLevelDataset.csv")
View(data)
df<-data
View(df)

str(df)
summary(df)
missing_values <- colSums(is.na(df))
print(missing_values)
table(df$stress_level)
df$stress_level=factor(df$stress_level,levels = c(0,1,2),labels = c(0,1,2))
sum(is.na(df))

df[-21]=scale(df[-21])
library(caTools)
set.seed(100)
sp=sample.split(df$stress_level,SplitRatio = 0.70)
traindf=subset(df,sp==T)
testdf=subset(df,sp==F)

#knn 
library(class) 
iknn=knn(train=traindf,test=testdf,cl=traindf$stress_level,k=25) 
library(caret) 
confusionMatrix(testdf$stress_level,iknn)



#naive bayes 
library(e1071) 
inb=naiveBayes(traindf[-21],traindf$stress_level) 
ipre=predict(inb,testdf[-21]) 
confusionMatrix(ipre,testdf$stress_level)

cm <- confusionMatrix(ipre, testdf$stress_level)

#decision tree 
library(rpart) 
idt=rpart(formula = stress_level~.,data=traindf) 
idpre=predict(idt,testdf[-21],type="class") 
confusionMatrix(idpre,testdf$stress_level)

library(rpart.plot)
rpart.plot(idt)


#randomForest 
library(randomForest) 
irf=randomForest(x=traindf[-21],y=traindf$stress_level,ntrees=25) 
irpre=predict(irf,testdf[-21]) 
confusionMatrix(irpre,testdf$stress_level)

varImpPlot(irf)


#svm 
library(e1071) 
u3 <- svm(stress_level ~ ., data = traindf, kernel = "linear", type = "C-classification")
pre=predict(u3,testdf) 
library(caret) 
confusionMatrix(pre,testdf$stress_level)


# Confusion Matrix
cm_df <- as.data.frame(as.table(cm))

ggplot(data = cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix",
       x = "Actual",
       y = "Predicted") +
  theme_minimal()

