
# [KT A.I. Web Novel Contest](https://blog.kt.com/1063)

![](https://github.com/hyungkwonko/novel-generator/blob/master/img/img1.jpg)

Host: [Korea Telecom](https://corp.kt.com/)

Result: 2nd place

This repository includes 3 APIs made in news big data hackathon in April 2019. Teams were assigned to develop a business model to create value using news big data. We made a web service platform that shows multi-Indicator of given keyword to understand how influential it is. We made a **[demo video](https://youtu.be/NUF3Wh3QoEs)**.

<br>

### [API 01. Sentiment Analysis](https://github.com/hyungkwonko/NewsBigDataAnalysis/tree/master/SentimentAnalysis)
![](https://github.com/hyungkwonko/NewsBigDataAnalysis/blob/master/img/pic2.png)

With the neural network architecture, the model learns whether the given sentence is positive or negative. Going through the grid search made it possible to find the best parameter for the model. After building this model, we made a histogram which shows the sentiment toward a single keyword.

<br>

### [API 02. Related Keywords](https://github.com/hyungkwonko/NewsBigDataAnalysis/tree/master/RelatedKeywords)
![](https://github.com/hyungkwonko/NewsBigDataAnalysis/blob/master/img/pic3.png)

There are keywords that matters for each individual. We tried to combine the data we get from news article and the feelings of people from one of the famous social media, Twitter. Since Twitter data is free to use for everyone, we were able to use it getting more insight. 

<br>

### [API 03. Article 2 Vector for clustering](https://github.com/hyungkwonko/NewsBigDataAnalysis/tree/master/A2V)
![](https://github.com/hyungkwonko/NewsBigDataAnalysis/blob/master/img/pic4.png)

![](https://github.com/hyungkwonko/NewsBigDataAnalysis/blob/master/img/pic5.png)

By optimizing the deep learning architecture, we made a new classifying algorithm, so-called Article 2 Vector. We developed this since there are so many similar news articles on the Internet. We hoped that going through the process of Article 2 Vector would reduce the meaningless web searching time looking for the information that one had seen before. After appropriate preprocessing, it has the ability to distinguish one from another. It can work for clustering, as the similar articles have similar vectors in the higher dimension. We classified them based on the cosine similarity between them.

