package skeleton;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Map.Entry;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import preprocess.HybridTokenizer;
/** 
* 功能描述：pefixspan序列模式挖掘算法,输入:序列集合,输出:最大频繁序列
* @created 2012-01-28 下午 2:42:54 
* @version 1.0.0 
*/
public class PrefixSpanBuild {
	private	static Logger log = LoggerFactory.getLogger(PrefixSpanBuild.class);
	private List<List<List<String>>>sequences =null; 	//序列集合
	private List<List<String>> maxFrSeqs = new ArrayList<List<String>>();  //最大频繁序列
	private List<String> itemList = new ArrayList<String>();//单项集合
	private int total = 0; //单项序列总和数
	private int minSup = 0; //最小支持数(默认两个) 
	private int minFrElemSize =0; //最小限制频繁序列元素数(默认两个) 
	private int maxFrElemSize =0;//最大限制频繁序列元素数(默认3个) 
	public PrefixSpanBuild(List<List<List<String>>>seqs)
	{
		this(seqs,2,2,3);
	}
    public PrefixSpanBuild(List<List<List<String>>>seqs,int minSup)
    {
    	this(seqs,minSup,2,3);
    }	
    public PrefixSpanBuild(List<List<List<String>>>seqs,int minSup,int minFrElemSize)
    {
    	this(seqs,minSup,minFrElemSize,3);
    }
    public PrefixSpanBuild(List<List<List<String>>>seqs,int minSup,int minFrElemSize,int maxFrElemSize)
    {
    	// 最小项集必须小于或等于限制项集数
    	if(minFrElemSize<=maxFrElemSize){
    		this.sequences = seqs;
    		this.minSup = minSup;
    		this.minFrElemSize = minFrElemSize;
    		this.maxFrElemSize = maxFrElemSize;
    		for(List<List<String>> elem : this.sequences){
    			for(List<String> items : elem){
    				for(String item: items){
    					if(!itemList.contains(item)){
    						itemList.add(item);
    						total++;
    					}
    				}
    			}
    		}
    	}
    }
    /**
    * 
    *  功能描述：计算每个单项的支持数
    * @return每个单项的支持数
    */
    protected Map<String,Integer>countFr1(){
    	log.info("开始读取每个单项的支持数");
    	Map<String,Integer> supMap = new LinkedHashMap<String,Integer>();
    	Integer sup = 0; //计算每个单项出现的次数（支持数）
    	Set<String> itemsSet = null;
    	for(List<List<String>> elem: sequences){
    		for(List<String>items : elem){
    			itemsSet = new HashSet<String>();
    			itemsSet.addAll(items);
    			for(String item: itemsSet){
    				if(itemList.contains(item)){
    					if(supMap.containsKey(item)){
    						sup = supMap.get(item)+1;    						
    					}else{
    						sup = 1;
    					}
    					supMap.put(item, sup);
    				}
    			}
    		}
    	}
    	for(Iterator<Entry<String,Integer>> iter = supMap.entrySet().iterator();iter.hasNext();){
    		Entry<String,Integer> supEntry = (Entry<String,Integer>) iter.next();
    		sup = supEntry.getValue();
    		if(sup<minSup){
    			iter.remove();
    		}
    	}
    	total = supMap.size();
    	log.info("读取完毕");
    	return supMap;
    }
    public List<String> replace(List<String>strList,String[] prefixSeq){
    	List<String> retainList = null;
    	int i = strList.size();
    	int length = prefixSeq.length;
    	int pla = strList.indexOf(prefixSeq[length - 1]);
    	if (pla >= 0 && pla < i - 1){
    		retainList = new ArrayList<String>();
    		if(length == 1){
    			retainList.add("_");
    		}else{
    			for(int k = 0;k <= pla;k++){
    				retainList.add("_");
    			}
    		}
    		retainList.addAll(strList.subList(pla+1,i ));
    	}
    	return retainList;
    }
    /**
    * 功能描述：temp_s在其投影数据库中查找再次出现它的次数
    * @param t_num 序列总数
    * @param temps 序列
    * @param sd[] 投影数据库
    * @param sd_count[]对应的索引
    * @return int
    */
    public int makeout(String temps,List<List<List<String>>>sdSeqs){
    	return makeout(new String[]{temps},sdSeqs);
    }
    /** 
    *  
    * 功能描述：temp_s 在其投影数据库中查找再次出现他的次数
    * @param tempSeq 序列
    * @paramsdSeqs   投影数据库
    */
    public int makeout(String[]tempSeq,List<List<List<String>>>sdSeqs){
    	String[] tempsSeqClone = tempSeq.clone(); 
    	int tMincout  =0;   
    	for(List<List<String>> sdElem :sdSeqs){
    		for(List<String>sdItems :sdElem){
    			int n = - 1  ;
    			n =  containArrays(sdItems,tempsSeqClone);
    			if(n >=0){
    				tMincout ++;
                    break;
    			}
    		}
    	}   
    	return tMincout;
    }
    /**
    * 功能描述：用prefixspan算法求出序列集的频繁序列
    */
    protected void prefixSpan(String[] prefixSeq,List<List<List<String>>>seqs,int prefixNum){
    	// 如果前缀得出的子序列的长度大于maxFrElemSize,则直接跳出
    	if(prefixNum > this.maxFrElemSize - 1){
    		return;
    	}
    	for(int tTotal = 0;tTotal < total;tTotal++){
    		// 第一种情况a的投影数据库seqs,循环整个单项集合ItemList,看是否存在某个item在seqs上还存在频繁单项eg:<a><b> 
    		int supNum1 = 0;
    		String tempSeq = itemList.get(tTotal);
    		supNum1 = makeout(tempSeq,seqs);

    		if(supNum1 >= minSup){
    			//开始记录频繁序列
    			List<String> itemList = new ArrayList<String>();
    			if(prefixNum >= this.minFrElemSize - 1){
    				for(int i=0;i<prefixNum;i++){
    					itemList.add(prefixSeq[i]);
    				}
    				itemList.add(tempSeq);
    				itemList.add(supNum1+"");//添加支持数
    				itemList.add((float)supNum1 / seqs.size()+"");//添加置信度
    				maxFrSeqs.add(itemList);
    			}
    			List<List<List<String>>> sdSeqs = generateSD(seqs,tempSeq);
    			String prefixSeq2[] = new String[prefixNum+1];
    			for(int e=0;e<prefixNum;e++)
    				prefixSeq2[e] = prefixSeq[e];
    			prefixSeq2[prefixNum] = tempSeq;
    			prefixSpan(prefixSeq2,sdSeqs,prefixNum+1);
    		}
    		//第二种情况a和ItemList的某个单项进行组合,看是否在seqs是还存在大于最小支持数的item eg:<a,b>
    		int supNum2 = 0;
    		String tempSeq1 = prefixSeq[prefixNum-1]+","+itemList.get(tTotal);
    		String tempSeq1s[] = tempSeq1.split(",");
    		supNum2 = makeout(tempSeq1s,seqs);
    		if(supNum2 >= minSup){
    			//开始记录频繁序列
    			List<String> itemList = new ArrayList<String>();
    			if(prefixNum >= this.minFrElemSize){
    				for(int i=0;i<prefixNum-1;i++){
    					itemList.add(prefixSeq[i]);
    				}
    				itemList.add(tempSeq1);
    				itemList.add(supNum2+"");//添加支持度
    				itemList.add((float)supNum2/seqs.size()+"");
    				maxFrSeqs.add(itemList);
    			}
    			List<List<List<String>>>sdSeqs = generateSD(seqs,tempSeq1s);
    			String aa[] = new String[prefixNum];
    			for(int e=0;e<prefixNum-1;e++)
    				aa[e] = prefixSeq[e];
    			aa[prefixNum-1] = tempSeq1;
    			prefixSpan(aa,sdSeqs,prefixNum);
    		}
    	}
    }
    public List<List<String>>buildPrefixSpan(){
    	Map<String,Integer>supMap = this.countFr1();
    	int times = 0;
    	log.info("符合支持度为{}，项集数为{}的总item数为{}",new Integer[]{minSup,minFrElemSize,total});
    	String itemId = null;
    	for(Entry<String,Integer>supEntry:supMap.entrySet()){
    		itemId = supEntry.getKey();
    		//生成投影数据库
    		List<List<List<String>>>sdList = this.generateSD(itemId);
    		String prefixSeq[] = {itemId};
    		this.prefixSpan(prefixSeq,sdList,1);
    		times++;
    		log.info("执行到itemId-{},已经循环到{}",new String[]{prefixSeq[0],times+""});
    	}
    	return this.maxFrSeqs;
    }
    
    public void printMaxFrSeq(){
    	StringBuffer tempStrBuf = null;
    	int seqSize = 0;
    	for(List<String>sequence:maxFrSeqs){
    		tempStrBuf = new StringBuffer();
    		seqSize = sequence.size();
    		tempStrBuf.append("<");
    		for(int i=0;i<seqSize-3;i++){
    			String skuId = sequence.get(i);
    			tempStrBuf.append(skuId+" ");
    		}
    		tempStrBuf.append(sequence.get(seqSize-3));
    		tempStrBuf.append(">");
    		tempStrBuf.append(" - "+sequence.get(seqSize-2));
    		tempStrBuf.append(" - "+sequence.get(seqSize-1));

        	System.out.println(tempStrBuf.toString());
    		log.info(((tempStrBuf.toString())));
    	}
    }
    /**
    * 根据前缀生成投影数据库
    * @param seqs 序列数据库 S
    */
    public List<List<List<String>>>generateSD(List<List<List<String>>>seqs,String prefixSeq){
    	return generateSD(seqs,new String[]{prefixSeq});
    }
    /**
    * 根据前缀生成投影数据库
    * @param prefixSeq 前缀
    */
    public List<List<List<String>>>generateSD(String prefixSeq){
    	return generateSD(sequences,new String[]{prefixSeq});
    }
    /**
    * 根据前缀生成投影数据库
    * @param seqs 序列数据库 S
    */
    public List<List<List<String>>>generateSD(List<List<List<String>>>seqs,String[] prefixSeq){
    	List<List<List<String>>> sdList = new ArrayList<List<List<String>>>();
    	List<String> retainItems = null;
    	List<List<String>> sdElem = null;
    	List<List<String>> retainsdElem = null;
    	for(List<List<String>> elem:seqs){
    		sdElem = new ArrayList<List<String>>();
    		for(List<String>item2s:elem){
    			int n = containArrays(item2s,prefixSeq);
    			if(n>=0){
    				retainItems = replace(item2s,prefixSeq);
    				if(retainItems!=null){
    					sdElem.add(retainItems);
    				}
    				retainsdElem = elem.subList(elem.indexOf(item2s)+1, elem.size());
    				if(!retainsdElem.isEmpty()){
    					sdElem.addAll(retainsdElem);
    				}
    				break;
    			}
    		}
    		if(!sdElem.isEmpty()){
    			sdList.add(sdElem);    		
    		}
    	}
    	return sdList;
    }
    /**
    * 功能描述：判断字符串中的数据是否在特殊集合(序列)中包含 eg:items=[a,b,c] temps=[_,_,c]是存在的"_"为任意字符串
    */
    public int containArrays(List<String>items,String[] temps){
    	int n = exitsArrays(items,temps);
    	if(n==-1&&temps.length>1){
    		String[] tempsClone = temps.clone();
    		for(int i=0;i<tempsClone.length-1;i++){
    			tempsClone[i] = "_";
    		}
    		n = exitsArrays(items,tempsClone);
    	}
    	return n;
    }
    /**
    * 功能描述：判断字符串中的数据是否在集合中存在
    */
    public int exitsArrays(List<String>items,String[] temps){
    	int n = -1;
    	int length = temps.length;
    	int size = items.size();
    	if(size>=length){
    		if(length==1 && !items.contains("_")){
    			n = items.indexOf(temps[0]);
    		}
    		if(length>1){
    			n = items.indexOf(temps[0]);//首先找到数组的第一个元素出现在集合中的位置
    			if(n+length<=size&&n!=-1){// 如果集合够长,且第一个存在
    				for(int i=1;i<length;i++){ //再按顺序找第二个到第N个
    					if(!temps[i].equals(items.get(i+n))){
    						n = -1;
    						break;
    					}
    				}
    			}else{
    				n = -1;
    			}
    		}
    	}
    	return n;
    }
    
    public static List<List<List<String>>> textPos(String inputPath)throws IOException{
    	Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);		

		List<List<List<String>>> sLists =new ArrayList<List<List<String>>>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("F:\\llt\\Data\\TweetDataset\\cluster\\"+inputPath.replace(".ids","\\6")), "utf-8"));
		String lineInfo=null;
	    while((lineInfo=br.readLine())!=null){
//	    	lineInfo = Eng_Similarity.getTokens(lineInfo);  //返回去掉标点符号，停用词的字符串
	    	
	    	HybridTokenizer hybridTokenizer = new HybridTokenizer();
		  	List<String> filteredTokenList = hybridTokenizer.tokenize(lineInfo);
		  	String[] TokenList=new String[filteredTokenList.size()];;
            filteredTokenList.toArray(TokenList);
            lineInfo = Arrays.toString(TokenList);
	    	
	    	
	    	List<String> text = new ArrayList<String>();
	    	List<List<String>> items =new ArrayList<List<String>>();
	    	Annotation document = new Annotation(lineInfo);
			pipeline.annotate(document);
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for(CoreMap sentence: sentences) {
				for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
//					String word = token.get(TextAnnotation.class);
					// this is the POS tag of the token
					String pos = token.get(PartOfSpeechAnnotation.class); //单词的POS
					String lemma = token.get(LemmaAnnotation.class);  //原型
//					text.add(pos);
					if(pos.equals("CD")) //数量词保留词性  （230，one TWO这样）
						text.add(pos);
					else
						text.add(lemma);
				}
			}			
	    	for(String tag: text){
	    		items.add(Arrays.asList(tag));  //后续代码需要的格式[word]
	    	}
	    	
//	    	for(String word:lineInfo.split(" ")) {
//	    		items.add(Arrays.asList(word));
//	    	}
	    	
	    	sLists.add(items);
	    }
	    br.close();	
	    return sLists;
    }
    public static void prefixSpan(List<List<List<String>>> sLists){
    	PrefixSpanBuild test = new PrefixSpanBuild(sLists,9,2,10);  //最小支持度，最短序列长度，最长序列长度
    	test.buildPrefixSpan();
    	System.out.println("序列数据库如下：");
    	for(List<List<String>>elem:test.sequences){
    		for(List<String>item2s:elem){
    			System.out.print(item2s);
    			System.out.print("      ");
    		}
    		System.out.println();
    	}
    	System.out.println("");
    	System.out.println("");
    	System.out.println("执行PrefixSpan算法，生成频繁序列模式结果如下：");
    	test.printMaxFrSeq();
    }
    
    public static void main(String[] args) throws IOException{
    	String inputPath = "./10";
    	List<List<List<String>>> sLists = textPos(inputPath);
    	prefixSpan(sLists);	
    }
   }
