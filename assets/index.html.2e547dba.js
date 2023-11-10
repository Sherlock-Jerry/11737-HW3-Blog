import{_ as o}from"./_plugin-vue_export-helper.cdc0426e.js";import{o as i,c as s,a as e,b as n,e as r,d as t,f as l,r as h}from"./app.9c4828c0.js";const c="/blog/assets/english-spanish.dec79b69.png",d="/blog/assets/english-xhosa.6218c270.png",m="/blog/assets/neural-network.0c659c39.png",u="/blog/assets/neural-machine-translation.cba05fb2.png",g="/blog/assets/siamese-network-diagram.8dfacee8.jpg",p="/blog/assets/training-translation-memory.11946590.png",f="/blog/assets/monolingual-translation-memory.07209f8e.jpg",w="/blog/assets/mtm-architecture.340b3c79.png",y={},b=e("p",null,[t("Hello fellow readers! In this post, I would like to share a recent advance in the field of Machine Translation. Specifically, I will be presenting the paper "),e("em",null,"Neural Machine Translation with Monolingual Translation Memory"),t(" by Cai et al, which received one of the six distinguished paper awards from ACL 2021.")],-1),v=t("Paper: "),_={href:"https://aclanthology.org/2021.acl-long.567/",target:"_blank",rel:"noopener noreferrer"},k=t("https://aclanthology.org/2021.acl-long.567/"),T=t(" Code: "),x={href:"https://github.com/jcyk/copyisallyouneed",target:"_blank",rel:"noopener noreferrer"},M=t("https://github.com/jcyk/copyisallyouneed"),q=e("h2",{id:"so-what-is-machine-translation",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#so-what-is-machine-translation","aria-hidden":"true"},"#"),t(" So... what is Machine Translation?")],-1),I=t("You can probably guess from the name: over the last few decades, researchers have tried to get computers (a.k.a. "),z=e("strong",null,"machines",-1),N=t(") to "),S=e("strong",null,"translate",-1),A=t(" between human languages. Their hard work has resulted in a plethora of products, like Google Translate and Microsoft Translator. I'm sure all of you have played with these tools before (if not, "),j={href:"https://translate.google.com/",target:"_blank",rel:"noopener noreferrer"},H=t("there's no time like the present"),E=t(") and have tried inputting various kinds of sentences. Their translation quality is quite impressive, and you can get near-native translations in many contexts (e.g. simple sentences between widely-spoken languages). However, you also may have noticed that some languages have better performance than others. This difference is due to the amount of usable translation data available (we'll go over what this means below). One of the key challenges being worked on today is to bridge this gap between "),L=e("em",null,"low-resource",-1),D=t(" and "),V=e("em",null,"high-resource",-1),W=t(" languages."),C=l('<figure><img src="'+c+'"><img src="'+d+'"><figcaption align="center" style="font-size:1vh;">The Spanish output seems to be accurate, but it might be possible to improve the Xhosa translation (disclaimer - I do not speak Xhosa). The word &quot;recurrent&quot; is being translated into &quot;oluqhubekayo&quot;, which means &quot;ongoing&quot;. In order to better capture the spirit of recurrent neural networks, however, a word closer to &quot;self-referencing&quot; might be more appropriate. This example is intended to illustrate the need for better low-resource language performance in jargon-heavy contexts. </figcaption></figure><h3 id="data-is-key" tabindex="-1"><a class="header-anchor" href="#data-is-key" aria-hidden="true">#</a> Data is Key</h3><p>In order to understand what translation data is and why it is crucial for a good machine translation (MT) system, we first need to understand how these systems work. All of the current state-of-the-art use special kinds of programs called <em>neural networks</em>, which are infamous for being able to approximate <em>any</em> mathematical function by looking at examples.</p><figure><img src="'+m+'"><figcaption align="center" style="font-size:1vh;">In this case, the network is learning to approximate f(x) = x + 1 just by looking at set of examples. The end goal to make correct guesses for unseen inputs, like passing 1.5 and getting 2.5. </figcaption></figure><p>If we can convert an English sentence into a list of numbers, and if we can do the same for a Spanish sentence, then in theory, it should be possible for the network to learn how to convert the numbers from one form into the other. And if we can train it on a large number of sentence pairs, it just might learn the grammatical rules well enough to provide good translations for unseen inputs.</p><figure><img src="'+u+'"><figcaption align="center" style="font-size:1vh;">Instead of a single number, the network is learning to map between entire sequences of numbers. This is still possible, and there are special network architectures optimized for &quot;sequence to sequence&quot; tasks. </figcaption></figure><p>Generally, hundreds-of-thousands or millions of parallel sentences are needed for good performance. However, it is hard to obtain pure, parallel datasets of this size for low-resource languages.</p><h3 id="monolingual-data" tabindex="-1"><a class="header-anchor" href="#monolingual-data" aria-hidden="true">#</a> Monolingual Data</h3><p>Even though low-resource languages like Xhosa may lack large amounts of parallel data, they still have vast amounts of untranslated text. If incorporated creatively, this <em>monolingual</em> (i.e. untranslated) text can also be used to help the network learn. Many strategies exist:</p><ul><li><em>Back-translation</em> uses an okay-performing, reversed translation model to turn each sentence from the monolingual data into a synthetic (fake) parallel pair. We can then train the main model on this new parallel data to hopefully expose it to a wider variety of sentences.</li><li>One could also use this data to <em>pre-train</em> the model before training on the parallel data. During the pre-training, all one has to do is delete random words from the monolingual corpus (e.g. a Xhosa book) and train the model to fill in the blanks. If the model does this task successfully and then trains on the parallel data, it may make better translations (at least, according to results published in <em>Multilingual Denoising Pre-training for Neural Machine Translation</em>).</li></ul><p>However, the main advance I will be sharing with you presents an entirely new way of using monolingual data. Specifically, it combines monolingual data with a mechanism called <em>Translation Memory</em>.</p><h3 id="translation-memory" tabindex="-1"><a class="header-anchor" href="#translation-memory" aria-hidden="true">#</a> Translation Memory</h3><p>Before getting into what <em>Translation Memory</em> actually is, let me first motivate it a little. Intuitively, a good-performing translation program should be able to perform well in a wide variety of contexts. Specifically, it should be able to translate sentences from speeches, news articles, folk tales, research papers, tweets, random blog posts on machine translation, etc. However, if you want a generic, jack-of-all-trades model to make passable translations in all of these areas, you inevitably have to make some sacrifices. This is where the <em>infrequent words</em> problem comes in: words that are infrequently-encountered by the model during training get discarded as noise, reducing performance in specialized domains. For example, it could forget the word &quot;chromosome&quot; and translate a biology textbook incorrectly. This happens to humans too. If you&#39;re trying to become an expert at 10 topics in a short timespan, you may easily forget technical words crucial to each of the 10 topics. If one of those topics is the <em>Medieval History of Mathematics</em>, you may easily forget who <em>al-Khawarizmi</em> was, causing you to mis-attribute his discoveries. (It&#39;s even harder for a neural network, as it could never exploit the mnemonic connection between <em>algorithm</em> and <em>al-Khawarizmi</em> \u{1F604}).</p><p>However, the good news is that computers can look things up extremely quickly, far more quickly than humans. Imagine if you were tested on the <em>Medieval History of Mathematics</em>, but you had the textbook&#39;s glossary with you. All of a sudden, if you were asked about <em>al-Khawarizmi</em>, you could provide the correct answer in the time that it takes you to look him up. <em>Translation Memory</em> essentially imbues the neural network with this same capability.</p><p>Essentially, each time the network is asked to generate a word, it can reference the translation memory (a bilingual dictionary mapping words between the source and target languages) to provide a more nuanced translation. The original researchers who proposed this concept came up with a two-component model architecture. One component consists of a neural network that generates its own &quot;guess&quot; for the translation word-by-word. The other component, called a memory component, retrieves the translations for each source word from a large dictionary. These two proposals are then combined, so that the one with higher confidence is used in the final translation.</p><p>(Note that everything in this section comes from <em>Memory-augmented Neural Machine Translation</em> [2]. If you want to learn more, go read their paper! I promise you, it&#39;s very interesting).</p><h3 id="end-of-intro" tabindex="-1"><a class="header-anchor" href="#end-of-intro" aria-hidden="true">#</a> End of Intro</h3><p>Thank you so much for bearing with me through the introduction. I did my best to put these topics into understandable words, but I may have accidentally glossed over something without explaining. If you have any questions or feedback, please don&#39;t hesitate to reach out to me at rajansaini@ucsb.edu.</p><h2 id="monolingual-translation-memory" tabindex="-1"><a class="header-anchor" href="#monolingual-translation-memory" aria-hidden="true">#</a> Monolingual Translation Memory</h2><p>Now, the moment we have all been waiting for has finally arrived. We can finally talk about what Monolingual Translation Memory is, how it exploits monolingual data, and what its implications are.</p><h3 id="intuition" tabindex="-1"><a class="header-anchor" href="#intuition" aria-hidden="true">#</a> Intuition</h3><p>The original Translation Memory introduced above is quite powerful, but it has a few limitations:</p><ul><li>The first is that it requires a parallel, bilingual dictionary. This means that when trying to come up with a translation for an unknown word, the model will try to translate it <em>directly</em> rather than use the entire sentence&#39;s context.</li><li>In addition, it is impossible for the retrieval mechanism itself to adapt as the model trains (a dictionary cannot change, even though other words might be more relevant).</li></ul><p>The new Monolingual Translation Memory is designed to solve both of these issues by:</p><ul><li>Using entire aligned sentences instead of a word-to-word dictionary. Such a sentence-sentence dictionary would be prohibitively long for humans to read through, but a clever retrieval mechanism would make it usable for a computer program. Furthermore, they use another neural network called a &quot;parallel encoder&quot; to determine whether two sentences are translations of each other. This allows them to associate monolingual sentences with existing translations, exploding the number of possibilities! (If you&#39;re confused by this, don&#39;t worry; this will be explained in more detail in a section below)</li><li>Making the retrieval mechanism <em>learnable</em>. This means that as the model trains on a parallel corpus, the retrieval mechanism should also be able to adapt itself. Specifically, it should learn to provide the most relevant translations from a large set of sentences (including sentences outside the original sentence-sentence dictionary).</li></ul><h3 id="parallel-encoders" tabindex="-1"><a class="header-anchor" href="#parallel-encoders" aria-hidden="true">#</a> Parallel Encoders</h3><p>The main secret behind this advance is its usage of <em>parallel encoders</em>. These are neural networks (i.e. mathematical functions) that map sentences to their meaning. More precisely, they map sentence embeddings (the original sentence converted to numbers, see &quot;Data is Key&quot;) to an <em>encoding vector</em>. The hope is for sentences with the <em>same meaning</em> to have the <em>same encoding</em>, even if they are expressed differently. For example, a good encoder would give &quot;I find recurrent neural networks fascinating&quot; and &quot;recurrent neural networks are fascinating to me&quot; similar encodings.</p><figure><img src="'+g+'"><figcaption align="center" style="font-size:1vh;">This concept being illustrated is also known as a Siamese network. They have been used successfully outside of machine translation, for tasks like handwritten signature verifiction and facial recognition. </figcaption></figure><p>We can also extend this idea across languages! We can have an encoder for each language that converts their sentences into a shared &quot;meaning space&quot;. More precisely, two sentences from different languages that share the same meaning should get mapped to similar encoding vectors:</p><p><img src="'+p+'" alt=""></p><p>Before we train our main neural network, we first train the target and source encoders on parallel text. The goal is for them to output identical encodings when they have the same meaning and different encodings when their meaning is different. This is called the &quot;alignment step&quot;. Then we can run the target encoder over a large untranslated corpus to encode each sentence. Then, every time the user wants a new sentence translated, we can find the target sentence with closest meaning by comparing the encodings:</p><p><img src="'+f+'" alt=""></p><p>Through this method, every sentence ever written in the target language can now be retrieved (at least in theory). The researchers that proposed this process found that searching for the most-similar encoding is equivalent to performing a Maximum Inner Product Search (MIPS). Fast algorithms that solve MIPS have already been discovered, so they can also be used for a speedy retrieval.</p><h3 id="main-translation-model" tabindex="-1"><a class="header-anchor" href="#main-translation-model" aria-hidden="true">#</a> Main Translation Model</h3><p>The retrieval model described above will give native translations, since it searches through text originally written by humans. However, what about completely new inputs for which direct translations have never existed? In that case, we would still want to use a larger neural network to create novel translations. If we could somehow allow that network to have access to some of the best-matched sentences found by the retrieval model, this should give us the best of both worlds. In highly domain-specific cases, the retrieval model could pull up relevant words or context that the network can use. The network could then produce a translation that matches the source sentence while taking advantage of these technical words.</p><figure><img src="'+w+'"><figcaption align="center" style="font-size:1vh;">This figure was taken from the original paper. The retrieval model is expected to output a list of sentence encodings (z1, z2, ...) and their similarities to the original input sentence (f(x,z1), f(x,z2), ...). This information is then fed into the main network (the Translation Model), along with the source sentence&#39;s encoding. </figcaption></figure>',36),P=t(`In order to pass the encodings and retrieved sentences to the translation model, we can use a memory encoder and an attention mechanism. The memory encoder is a simple learnable matrix that maps each sentence to a new vector space (this section is a little more advanced, but "learnable" means that the matrix's weights will get adjusted during training) . I would guess that this is done so that the source sentence and retrieved sentences get mapped to the same vector space. This way, they can be meaningfully compared with and added to each other. After the retrieved sentences get transformed into memory embeddings, an attention mechanism combines them with the source sentences (scaled back by their confidences). I will not explain the full details behind how the attention mechanism works (`),B={href:"https://towardsdatascience.com/the-intuition-behind-transformers-attention-is-all-you-need-393b5cfb4ada",target:"_blank",rel:"noopener noreferrer"},F=t("this article"),K=t(" has a great explanation), but the intuitive idea is that it highlights the relevant items from the memory encoder based on the source embeddings. After that, a decoder network converts the attention scores into the final translation, word-by-word."),Z=e("h2",{id:"done-at-last",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#done-at-last","aria-hidden":"true"},"#"),t(" Done! At last")],-1),O=e("p",null,"Whew! That was a lot! Anyway, this is my possibly-confusing attempt at sharing one of the latest advances in Machine Translation. If you need any clarification (especially with the translation model above), definitely feel free to reach out! Otherwise, thank you so much for your patience to have made it this far.",-1),X=e("h2",{id:"references",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#references","aria-hidden":"true"},"#"),t(" References")],-1),Y=e("p",null,"[1] Deng Cai, Yan Wang, Huayang Li, Wai Lam, Lemao Liu. Neural Machine Translation with Monolingual Translation Memory. ACL 2021.",-1),G=e("p",null,"[2] Yang Feng, Shiyue Zhang, Andi Zhang, Dong Wang, Andrew Abel. Memory-augmented Neural Machine Translation. EMNLP 2017.",-1),R=e("p",null,[t("[3] Victor Zhou. \u201CNeural Networks from Scratch.\u201D "),e("em",null,"Victor Zhou"),t(", Victor Zhou, 9 Feb. 2020, https://victorzhou.com/series/neural-networks-from-scratch/.")],-1);function U(J,Q){const a=h("ExternalLinkIcon");return i(),s("div",null,[b,e("p",null,[v,e("a",_,[k,n(a)]),T,e("a",x,[M,n(a)])]),r("- more -"),q,e("p",null,[I,z,N,S,A,e("a",j,[H,n(a)]),E,L,D,V,W]),C,e("p",null,[P,e("a",B,[F,n(a)]),K]),Z,O,X,Y,G,R])}const te=o(y,[["render",U],["__file","index.html.vue"]]);export{te as default};
