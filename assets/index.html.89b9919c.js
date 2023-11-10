import{_ as o}from"./_plugin-vue_export-helper.cdc0426e.js";import{o as i,c as r,e as s,a as e,b as n,f as h,d as t,r as d}from"./app.9c4828c0.js";const c="/blog/assets/ALPAC.914e1198.png",l="/blog/assets/bleu_exm.47689ec3.png",m="/blog/assets/bert_tech.1630e3ef.png",u="/blog/assets/bert_e1.1b12ee1a.png",p="/blog/assets/bert_e2.8ddac111.png",f="/blog/assets/bert_exm.efd89c24.png",g="/blog/assets/bert_t1.fa80de21.png",b="/blog/assets/bert_t4.a1925c45.png",w="/blog/assets/comet_t1.4e9391cc.png",v="/blog/assets/comet_t2.91a918c9.png",y="/blog/assets/comet_t3.cb3bfb08.png",T="/blog/assets/comet_t4.2ed63c43.png",_="/blog/assets/comet_t5.08631351.png",E="/blog/assets/comet_t6.50fef4b7.png",x="/blog/assets/comet_e1.66f4dadd.png",M="/blog/assets/comet_e2.0e1ff4bb.png",R="/blog/assets/comet_e3.7afe949c.png",S={},B=e("p",null,"How to automatically evaluate the quality of a machine translation system? Human evaluation is accurate, but expensive. It is not suitable for MT model development.",-1),k=e("p",null,"Reading Time: About 15 minutes.",-1),A=h('<h2 id="a-brief-history-of-mt-evaluation-metrics" tabindex="-1"><a class="header-anchor" href="#a-brief-history-of-mt-evaluation-metrics" aria-hidden="true">#</a> A brief history of MT evaluation metrics</h2><h3 id="human-evaluation" tabindex="-1"><a class="header-anchor" href="#human-evaluation" aria-hidden="true">#</a> Human evaluation</h3><p>In 1966, United States, the Automatic Language Processing Advisory Committee (ALPAC) conducted a large scale study on the evaluation of the state-of-the-art Russian-to-English Machine Translation (MT) systems at that time [1]. Indeed, the ALPAC report was infamous for holding a negative opinion toward the development of MT, and caused the suspension of research into related fields for two decodes. However, one of the first practical method for the evaluation of translation quality was developed from the study. Basically, six trained translators were each assigned to evaluate 144 sentences from 4 passages. The evaluation was based on &quot;intelligibility&quot; and &quot;fidelity&quot;. &quot;Intelligibility&quot; measures to what extent the sentence can be understood, and &quot;fidelity&quot; measures how much information the translated sentence retained compared to the source. Human evaluation was based on these two variables by giving a score on the scale of 1-9. This is one of the earlest systematic MT evaluation metrics based on human judgement. <img src="'+c+'" alt="image1"></p><h3 id="automatic-evaluation" tabindex="-1"><a class="header-anchor" href="#automatic-evaluation" aria-hidden="true">#</a> Automatic evaluation</h3><p>Even though employing human judgement as measuring metric is the most effective approach, purely depending on human is expensive as well as slow in face of the growing size of data, which promoted the need for automation. In 2002, the most commonly used evaluation metric, Bilingual Evaluation Understudy (BLEU), was developed by Kishore et al. [2]. BLEU measures the difference between references and machine translation candidates through n-grams and brevity penalty. Based on the preliminary that the \u201Chighest correlation with monolingual human judgements\u201D is four, n-grams measure the exact word segment correspondence of length one to four in the sentence pair. The brevity penalty is included to avoid short candidates receiving unreasonably high BLEU scores. BLEU remains popular till today due to its light-weightedness and fastness. A simple example [3] of word-level BLEU is demonstrated below. <img src="'+l+'" alt="image2"></p><h2 id="bertscore" tabindex="-1"><a class="header-anchor" href="#bertscore" aria-hidden="true">#</a> BERTScore</h2><p>Recent works on MT quality evaluation have provided stronger metrics and supports to the increased research interest in neural methods for training MT models and systems. BERTScore, which appeared in the 2020 International Conference on Learning Representations, aims to develop \u201Can automatic evaluation metric for text generation\u201D [4]. As a high level summary, BERTScore is one step forward from the commonly used BLEU, because BERTScore incorporates the additional contextual information into consideration to calculate the degree of difference between source and target sentence.</p><h3 id="motivation" tabindex="-1"><a class="header-anchor" href="#motivation" aria-hidden="true">#</a> Motivation</h3><p>Generally speaking, there are two drawbacks in the n-gram-based metrics. Firstly, semantically-correct translations or paraphrases are excessively penalized for the n-gram metrics. In other words, different usage of words on the surface level will result in a low BLEU score. In the paper, the authors give the example of the source reference sentence \u201Cpeople like foreign cars,\u201D and two of the candidates are \u201Cpeople like visiting places abroad\u201D and \u201Cconsumers prefer imported cars.\u201D The latter uses synonyms to replace certain words in the reference, while preserving the original semantic meanings. However, n-gram-based metrics like BLEU will give higher score to the former candidate, even though the meaning is far from that of the reference sentence, since the exact string match of unigram and bigram values are higher. In face of this pitfall, the BERTScore authors are motivated to break the restrictions of n-grams, and to take advantage of contextualized token embedding as the matching metric, by calculating cosine similarities of all pairs in the reference and candidate.</p><p>Secondly, n-gram metrics cannot capture semantic dependencies of distant words or penalize semantically-critical word order changes. For example, for short sentences, BLEU is able to capture the swap of cause and effect clauses, like \u201CA results in B.\u201D However, when A and B are long phrases, even the longest four-gram will fail to capture the cause-effect semantic dependencies of A and B if their order change. The n-gram metrics measures the similarity in a shallow way, which motivates the authros to develop a metric that is more effective in tackling the distant dependencies and ordering problems.</p><h3 id="technique" tabindex="-1"><a class="header-anchor" href="#technique" aria-hidden="true">#</a> Technique</h3><p>The workflow of BERTScore computation is illustrated in the diagram below. Having a reference sentence x tokenized to (x1, \u2026, xk) and a candidate sentence x\u0302 tokenized to (x\u03021, ..., x\u0302k), the technique transforms the tokens into contextual imbeddings, and compute the match among all takens by cosine similarity. As an option, multiplying an additional weight based on the inverse document frequency of matching words can be helpful in some scenarios. The outcome includes a precision (R_BERT), recall (P_BERT), and combined metric scores(F1). <img src="'+m+'" alt="image3"></p><p>BERTScore uses the BERT model to generate contextual embeddings for each token. BERT tokenizes the input text into a sequence of word pieces, and splits the unknown words into commonly observed sequences of characters. The Transformer encoder computes the representation for each word piece by repeatedly applying self-attention and nonlinear transformation alternatively. The resulting contextual embedding from word piece will generate different vector representation for the same word piece in different contexts with regard to surrounding words, which is significantly different from the exact string match metric in BLEU.</p><p>Due to the vector representation of word embedding, BERTScore is able to perform a soft measure of similarity compared to exact-string matching in BLEU. The cosine similarity of a reference token xi and a candidate token x\u0302j is : <img src="'+u+'" alt="image4"></p><p>With similarity measurement of each pair of reference token and candidate token in preparation, we can move on to compute precision and recall. In the greedy match perspective, we match each token in x with the highest similarity score in x\u0302.Recall is computed by matching each token in x to a token in x\u0302, while precision is by matching each token in x to the corresponding token in x\u0302. F1 score is calculated by combining precision and recall with the formular listed below. Extensive experiments indicate that F1 score performs reliably well across different settings, and therefore is the most recommended score to be used for evaluation. <img src="'+p+'" alt="image5"></p><p>Optionally, we can add an importance weighting to different words to optimize the metric, because previous works indicated that \u201Care words can be more indicative for sentence similarity than common words\u201D [5]. From experiments, apply idf-based weight can render small benefits in some scenarios, but have limited contribution in other cases. The authors use the inverse document frequency (idf) scores to assign higher weights to rare words. Because there is limited preformance improvement when applying importance weighting, details about this optional stage will not be discussed further.</p><p>A simple example of BERTScore calculation without importance from the ref-cand cosine similarity matrix is illustrated below. Basically, R_BERT is calculated by the sum of maximum values in each row divided by the number of rows, and P_BERT is calculated by the sum of the maximum values in each column divided by the number of columns. F1 is computed by 2 times the product of R_BERT and P_BERT divided by their sum. The BERTScore with importance weighting can be computed by multiplying the corresponding weight to each cosine similarity. <img src="'+f+'" alt="image6"></p><h3 id="effectiveness" tabindex="-1"><a class="header-anchor" href="#effectiveness" aria-hidden="true">#</a> Effectiveness</h3><p>For the evaluation of BERTScore, this blog will focus on the machine translation task in the original paper. The experiment\u2019s main evaluation corpus is the WMT18 metric evaluation dataset, containing predictions of 149 translation systems across 14 language pairs, gold references, and two types of human judgment scores. The evaluation is completed with regard to both segment-level and system-level. The Segment-level human judgment score is for each reference-candidate pair, while the system-level human judgments score is based on all pairs in the test set.</p><p>Table below demonstrates the system-level correlation to human judgements. The higher the score is, the closer the system evaluation is to human evaluation. Focusing on FBERT score (F1 score), we can see a large number of bold correlations of metrics for FBERT, indicating it is the top performance system compared to the others. <img src="'+g+'" alt="image7"></p><p>Apart from system-level correlation, the table below illustrating the segment-level correlations, BERTScore shows a considerably higher performance compared to the others. The outperformance in segment-level correlations further exhibits the quality of BERTScore for sentence level evaluation. <img src="'+b+'" alt="image8"></p><h2 id="comet" tabindex="-1"><a class="header-anchor" href="#comet" aria-hidden="true">#</a> COMET</h2><p>In 2020, Rei et al. presented \u201Ca neural framework for training multilingual machine translation evaluation models which obtains new state-of-the-art levels of correlation with human judgements\u201D at the 2020 Conference of Empirical Methods in Natural Language Processing [6]. The system, COMET, employs a different approach in improving evaluation metric. COMET builds an additional regression model to exploit information from source, hypothesis, and reference embeddings, and training the model to give a prediction on the quality of translation that highly correlates with human judgement.</p><h3 id="motivation-1" tabindex="-1"><a class="header-anchor" href="#motivation-1" aria-hidden="true">#</a> Motivation</h3><p>As the authors point out, \u201Cthe MT research community still relies largely on outdated metrics and no new, widely-adopted standard has emerged\u201D. This creates motivation for a metric scheme that uses a network model to actually learn and predict how well a machine translation will be in a human rating perspective. We knew that BLEU transformed MT quality evaluation from human rating to automated script, BERTScore improved the evaluation scheme by incoporating context, whereas COMET is motivated to learn how human will evaluate the quality of the translation, specifically scores from direct assessment (DA), human-mediated translation edit rate (HTER), and metrics compliant with multidimensional quality metric framework (MQM). After all, humans are the best to evaluate the translation quality of our own language. In short, COMET aims at closing the gap between automated metric with actual human evaluation.</p><h3 id="technique-1" tabindex="-1"><a class="header-anchor" href="#technique-1" aria-hidden="true">#</a> Technique</h3><p>The first step of COMET score computation is to encode the source, MT hypothesis, and reference sentence into token embeddings. The authors take advantage of a pretrained, cross-lingual encoder model, XLM_RoBERTa, to generate the three sequences (src, hyp, ref) into token embeddings. For each input sequence x = [x0, x1, \u2026, xn], the encoder will produce an embedding e_j(l) for each token xj and each layer l \u2208 {0, 1, \u2026, k}.</p><p>The word embeddings from the last layer of the encoders are fed into a pooling layer. Using a layer-wise attention mechanism, the information from the most important encoder layers are pooled into a single embedding for each token ej. \u03BC is a trainable weight coefficient, E_j = [e_j(0), e_j(1), ..., e_j(k)] corresponds to the vector of layer embeddings for token xj, and \u03B1 = softmax([\u03B1(1), \u03B1(2), . . . , \u03B1(k)]) is a vector corresponding to the layer-wise trainable weights. <img src="'+w+'" alt="image9"></p><p>After applying an average pooling to the resulting word embeddings, a sentence embedding can be concatenated into a single vector from segments. The process is repeated three times for source, hypothesis, and reference sequences. Specifically, two models, the Estimator model and the Translation Ranking model, were developed for different usages.</p><p>For the Estimator model, a single vector x is computed from the three sentence embeddings s, h, and r specified below: <img src="'+v+'" alt="image10"></p><p>Where h\u2299s and h\u2299r denotes the element-wise source product and reference product, and |h-s| and |h-r| denotes the absolute element-wise source difference and reference difference. The combined feature x serves as input to a feed-forward regression network. The network is trained to minimize the mean squared error loss between its predicted scores and human quality assessment scores (DA, HTER or MQM).</p><p>The Translation Ranking model, on the other hand, has different inputs {s,h+,h-,r}, i.e. a source, a higher-ranked hypothesis h+, a lower-ranked hypothesis h-, and a reference. After transforming them into sentence embeddings <strong>{s,h+,h-,r}</strong>, the triplet margin loss in relation to the source and reference is calculated: <img src="'+y+'" alt="image11"></p><p>d(u, v) denotes the euclidean distance between u and v and \u03B5 is a margin. In this way during training, the model will optimize the embedding space so the distance between the anchors (s and r) and the \u201Cworse\u201D hypothesis h\u2212 is larger by at least \u03B5 than the distance between the anchors and \u201Cbetter\u201D hypothesis.</p><p>In the inference stage, the model will receive a triplet input (s,\u0125,r) with only one hypothesis, and the quality score will be the harmonic mean between the distance to the source d(s,\u0125) and that to the reference d(r,\u0125), and normalized it to a 0 to 1 range: <img src="'+T+'" alt="image12"><img src="'+_+'" alt="image13"></p><p>In short, the Translation Ranking model is trained to minimize the distance between a \u201Cbetter\u201D hypothesis and both its corresponding reference and its original source. <img src="'+E+'" alt="image14"></p><h3 id="effectiveness-1" tabindex="-1"><a class="header-anchor" href="#effectiveness-1" aria-hidden="true">#</a> Effectiveness</h3><p>To test the effectiveness of COMET, the authors trained 3 MT translations models that target different types of human judgment (DA, HTER, and MQM) from the corresponding datasets: the QT21 corpus, the WMT DARR corpus, and the MQM corpus. Two Estimator models and one Translation Ranking model are trained. One regressed on HTER (COMET-HTER) is trained with the QT21 corpus, and another model regressed on MQM (COMET-MQM) is trained with the MQM corpus. COMET-RANK is trained with the WMT DARR corpus. The evaluation method employed is the official Kendall\u2019s Tau-like formulation. Concordant is the number of times the metric gives a higher score to the defined &quot;better&quot; hypothesis, while Discordant is the number of times the metrics give a higher score to the &quot;worse&quot; hypothesis or the scores is the same for the two hypothesis. <img src="'+x+'" alt="image15"></p><p>As shown in table the first table below, for as much as seven in eight language pair evaluation with English as source, COMET-RANK outperforms all other evaluation systems, including BLEU, two encoder models of BERTScore, and its two Estimator models, to a large extent. Similarly, for the language pair evaluation with English as target in the second table below, COMET also exceeds the other metrics in the overall performance, including the 2019 task winning metric YISI-1. <img src="'+M+'" alt="image16"><img src="'+R+'" alt="image17"></p><h2 id="case-study" tabindex="-1"><a class="header-anchor" href="#case-study" aria-hidden="true">#</a> Case Study</h2><p>In order to measure how well BLEU, BERTScore, and COMET can evaluate on existing MT systems, I managed to find a dataset with human judgment scores (e.g DA) [7]. Unfortunately, the MT systems that have the DA score is not available to the public, e.g. I cannot access the Baidu-system.6940 with the highest DA score in WMT19. With this preliminary, the experiment to compare how our evaluation metrics scores with a human judgement score (e.g. DA) is unattainable. Another simpler case study for the metrics is initialized instead.</p><p>For the setup, a group of 10 source-reference sentence pairs were prepared from a Chinese-English parallel Yiyan corpus [8]. The source Chinese sentences are fed to two common NMT systems: Google translate which uses Google Neural Machine Translation (GNMT) [9] and SYSTRAN translate [10], and the output of translation is stored as their corresponding hypthesis.</p><p>For BERTScore, we use the encoder from roberta without the importance weighting, and F1 score to evaluate the translated hypothesis. For COMET, we use the reference-free Estimation model \u201Cwmt20-comet-qe-da\u201D, trained based on DA and used Quality Estimation (QE) as a metric, for the evaluation on GNMT and SYSTRAN. The scores from BLEU, BERTScore, and COMET are illustrated in the table below. With the limited 10 data samples, BERTScore and COMET consider Google Translator performing better, while the BLEU score for SYSTRAN Translator is higher.</p><table><thead><tr><th>System-level score</th><th>Google</th><th>SYSTRAN</th></tr></thead><tbody><tr><td>BLEU</td><td>33.96</td><td>37.60</td></tr><tr><td>BERTScore F1</td><td>0.7934</td><td>0.7562</td></tr><tr><td>COMET</td><td>0.7215</td><td>0.6418</td></tr></tbody></table><p>The limitation of BLEU as compared to BERTScore and COMET is mostly exposed in the second sentence, as illustrated in the table below. The BLEU score for Google is 19.29, while that of SYSTRAN is 44.96. Though there is no DA scores from experts, the meanings of the two hypothesis and the reference are very similar, and the difference mostly lies on the different choice of same-meaning words. The n-gram&#39;s measurement based on the exact string match causes the large difference in the evaluation result. In comparison, the context-based BERTScore and human-judgement-trained COMET do not have a significant difference in their scores, and this example suggests the outdatedness of n-gram-based metrics to some extent.</p><table><thead><tr><th>Type</th><th>Sentence</th></tr></thead><tbody><tr><td>Src</td><td>\u6211\u4EEC\u5728\u7F51\u7EDC\u641C\u7D22\u548C\u5E7F\u544A\u7684\u521B\u65B0\uFF0C\u5DF2\u4F7F\u6211\u4EEC\u7684\u7F51\u7AD9\u6210\u4E3A\u5168\u4E16\u754C\u7684\u9876\u7EA7\u7F51\u7AD9\uFF0C\u4F7F\u6211\u4EEC\u7684\u54C1\u724C\u6210\u5168\u4E16\u754C\u6700\u83B7\u8BA4\u53EF\u7684\u54C1\u724C\u3002</td></tr><tr><td>Ref</td><td>Our innovations in web search and advertising have made our web site a top internet property and our brand one of the most recognized in the world.</td></tr><tr><td>Hyp_Google</td><td>Our innovation in online search and advertising has made our website a top website in the world, and our brand has become the most recognized brand in the world.</td></tr><tr><td>Hyp_SYSTRAN</td><td>Our innovations in online search and advertising have made our website the world&#39;s top website and made our brand the most recognized in the world.</td></tr></tbody></table><table><thead><tr><th>Segment-level score for 2nd sentence</th><th>Google</th><th>SYSTRAN</th></tr></thead><tbody><tr><td>BLEU</td><td>19.29</td><td>44.96</td></tr><tr><td>BERTScore F1</td><td>0.7515</td><td>0.7820</td></tr><tr><td>COMET</td><td>0.7399</td><td>0.7396</td></tr></tbody></table><p>Let\u2019s take a closer look at the 8th sentence shown below. Because the SYSTRAN&#39;s translation exactly matched the reference sentence, BLEU for this sentence is 100. In BERTScore, SYSTRAN also receives a score 0.2 higher than GNMT, because the former&#39;s translation matched more with the reference. However, we can clearly see that the result from Google Translate matches more with the source sentence in Chinese, especially the choice of word of \u201Cregistered\u201D instead of \u201Cincorporated\u201D for &quot;\u6CE8\u518C&quot;, and \u201CDelaware, USA\u201D instead of \u201CDelaware\u201D for &quot;\u7F8E\u56FD\u7279\u62C9\u534E\u5DDE&quot;. The COMET score for this sentence is 0.5144 for GNMT versus 0.3090 for SYSTRAN, which correlates more with human judgement. This is because COMET does not take the reference sentences but the source sentences in Chinese as input. COMET aims to mimic how human judgement (DA under this experimental setup) evaluates the translation, and clearly the Google translation provides a more exact translation from source. This example can be used to illustrate the limitation of metrics that purely depend on the reference sentence.</p><table><thead><tr><th>Type</th><th>Sentence</th></tr></thead><tbody><tr><td>Src</td><td>\u6211\u4EEC\u4E8E1998\u5E749\u6708\u5728\u52A0\u5229\u798F\u5C3C\u4E9A\u5DDE\u6CE8\u518C\u6210\u7ACB 2003\u5E748\u6708\u5728\u7F8E\u56FD\u7279\u62C9\u534E\u5DDE\u91CD\u65B0\u6CE8\u518C\u3002</td></tr><tr><td>Ref</td><td>We were incorporated in California in September 1998 and reincorporated in Delaware in August 2003.</td></tr><tr><td>Hyp_Google</td><td>We were registered in California in September 1998 and re-registered in Delaware, USA in August 2003.</td></tr><tr><td>Hype_SYSTRAN</td><td>We were incorporated in California in September 1998 and reincorporated in Delaware in August 2003.</td></tr></tbody></table><table><thead><tr><th>Segment-level score for 8th sentence</th><th>Google</th><th>SYSTRAN</th></tr></thead><tbody><tr><td>BLEU</td><td>37.06</td><td>100</td></tr><tr><td>BERTScore F1</td><td>0.7948</td><td>1.0000</td></tr><tr><td>COMET</td><td>0.5144</td><td>0.3090</td></tr></tbody></table><p>Not a trained translator myself, I cannot give my personal judgements on GNMT and SYSTRAN, but through the two examples, we clearly see the limitation of BLEU, and the limitation of BERTScore to some extent. However, it is still debatable if reference sentences should be evaluated in the metric. For COMET, inferring human judgement directly from source is appealing, but free-of-reference may result in loss of information in certain perspectives. Considering the experimental results has proven its effectiveness compared to BLEU and BERTScore, COMET may have pointed another direction for future MT evaluation metrics.</p><h2 id="conclusion" tabindex="-1"><a class="header-anchor" href="#conclusion" aria-hidden="true">#</a> Conclusion</h2><p>To sum up, two more advanced MT metric, BERTScore and COMET, are introduced. BERTScore enriches the information used in evaluation by incorporating contextual embedding to compute the degree of difference, and COMET employs an additional regression model to exploit information to make prediction score that correlates with human judgement. Walking through the history of MT metrics, we start from the most labor-intensive human evaluation, move a step further to automated n-gram-based metrics like BLEU, develop further on taking contextual information into consideration in BERTScore, and finally arrive at training models to evaluate like human in COMET. The development is exciting, but it is also worth noted that comparing to the recent dramatic improvement in MT quality, MT evaluation has fallen behind. In 2019, the WMT News Translation shared Task has 153 submissions, while the Metrics Shared Task only has 24 submissions [6]. The importance of MT evaluation should be the same as the MT techniques. With more advanced evaluation metrics to support and give feedbacks to new MT systems, the future development of MT realm as a whole can prosper.</p><h2 id="code" tabindex="-1"><a class="header-anchor" href="#code" aria-hidden="true">#</a> Code</h2>',53),C=t("BERTScore: "),L={href:"https://github.com/Tiiiger/bert_score",target:"_blank",rel:"noopener noreferrer"},q=t("https://github.com/Tiiiger/bert_score"),O=t(" COMET: "),N={href:"https://github.com/Unbabel/COMET",target:"_blank",rel:"noopener noreferrer"},U=t("https://github.com/Unbabel/COMET"),j=e("h2",{id:"reference",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#reference","aria-hidden":"true"},"#"),t(" Reference")],-1),F=e("p",null,"[1] ALPAC (1966) Languages and machines: computers in translation and linguistics. A report by the Automatic Language Processing Advisory Committee, Division of Behavioral Sciences, National Academy of Sciences, National Research Council. Washington, D.C.: National Academy of Sciences, National Research Council, 1966. (Publication 1416.)",-1),I=e("p",null,"[2] Kishore Papineni, Salim Roukos, Todd Ward, and Wei- Jing Zhu. Bleu: a method for automatic eval- uation of machine translation. ACL 2002.",-1),D=t('[3] Lei Li. "Data, Vocabulary and Evaluation," page 35-36. 2021. '),W={href:"https://sites.cs.ucsb.edu/~lilei/course/dl4mt21fa/lecture2evaluation.pdf",target:"_blank",rel:"noopener noreferrer"},G=t("https://sites.cs.ucsb.edu/~lilei/course/dl4mt21fa/lecture2evaluation.pdf"),H=t("[4] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. \u201CBERTScore: Evaluating Text Generation with BERT,\u201D ICLR 2020. "),Y={href:"https://openreview.net/forum?id=SkeHuCVFDr",target:"_blank",rel:"noopener noreferrer"},z=t("https://openreview.net/forum?id=SkeHuCVFDr"),P=t("."),Q=e("p",null,"[5] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for mt evaluation with improved correlation with human judgments. In IEEvaluation@ACL, 2005.",-1),K=t("[6] Rei, Ricardo, Craig Stewart, Ana C Farinha, and Alon Lavie. \u201CCOMET: A Neural Framework for MT Evaluation.\u201D EMNLP, 2020. "),V={href:"https://doi.org/10.18653/v1/2020.emnlp-main.213",target:"_blank",rel:"noopener noreferrer"},Z=t("https://doi.org/10.18653/v1/2020.emnlp-main.213"),J=t("."),X=t('[7] marvinKaster, "global-explainability-metrics," 2021. '),$={href:"https://github.com/SteffenEger/global-explainability-metrics/blob/main/WMT19/DA-syslevel.csv",target:"_blank",rel:"noopener noreferrer"},ee=t("https://github.com/SteffenEger/global-explainability-metrics/blob/main/WMT19/DA-syslevel.csv"),te=t('[8] Corpus Research Group, Beijing Foreign Studies University Foreign Language. "Yiyan English-Chinese Parallel Corpus," 2020. '),ae={href:"http://corpus.bfsu.edu.cn/info/1082/1693.htm",target:"_blank",rel:"noopener noreferrer"},ne=t("http://corpus.bfsu.edu.cn/info/1082/1693.htm"),oe=e("p",null,"[9] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google\u2019s neural machine translation system: Bridging the gap between human and machine translation.",-1),ie=e("p",null,"[10] Guillaume Klein, Dakun Zhang, Clement Chouteau, Josep M Crego, and Jean Senellart. 2020. Efficient and high-quality neural machine translation with opennmt. In Proceedings of the Fourth Workshop on Neural Generation and Translation.",-1);function re(se,he){const a=d("ExternalLinkIcon");return i(),r("div",null,[B,k,s(" more "),A,e("p",null,[C,e("a",L,[q,n(a)]),O,e("a",N,[U,n(a)])]),j,F,I,e("p",null,[D,e("a",W,[G,n(a)])]),e("p",null,[H,e("a",Y,[z,n(a)]),P]),Q,e("p",null,[K,e("a",V,[Z,n(a)]),J]),e("p",null,[X,e("a",$,[ee,n(a)])]),e("p",null,[te,e("a",ae,[ne,n(a)])]),oe,ie])}const le=o(S,[["render",re],["__file","index.html.vue"]]);export{le as default};
