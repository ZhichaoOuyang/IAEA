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
* ����������pefixspan����ģʽ�ھ��㷨,����:���м���,���:���Ƶ������
* @created 2012-01-28 ���� 2:42:54 
* @version 1.0.0 
*/
public class PrefixSpanBuild {
	private	static Logger log = LoggerFactory.getLogger(PrefixSpanBuild.class);
	private List<List<List<String>>>sequences =null; 	//���м���
	private List<List<String>> maxFrSeqs = new ArrayList<List<String>>();  //���Ƶ������
	private List<String> itemList = new ArrayList<String>();//�����
	private int total = 0; //���������ܺ���
	private int minSup = 0; //��С֧����(Ĭ������) 
	private int minFrElemSize =0; //��С����Ƶ������Ԫ����(Ĭ������) 
	private int maxFrElemSize =0;//�������Ƶ������Ԫ����(Ĭ��3��) 
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
    	// ��С�����С�ڻ�����������
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
    *  ��������������ÿ�������֧����
    * @returnÿ�������֧����
    */
    protected Map<String,Integer>countFr1(){
    	log.info("��ʼ��ȡÿ�������֧����");
    	Map<String,Integer> supMap = new LinkedHashMap<String,Integer>();
    	Integer sup = 0; //����ÿ��������ֵĴ�����֧������
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
    	log.info("��ȡ���");
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
    * ����������temp_s����ͶӰ���ݿ��в����ٴγ������Ĵ���
    * @param t_num ��������
    * @param temps ����
    * @param sd[] ͶӰ���ݿ�
    * @param sd_count[]��Ӧ������
    * @return int
    */
    public int makeout(String temps,List<List<List<String>>>sdSeqs){
    	return makeout(new String[]{temps},sdSeqs);
    }
    /** 
    *  
    * ����������temp_s ����ͶӰ���ݿ��в����ٴγ������Ĵ���
    * @param tempSeq ����
    * @paramsdSeqs   ͶӰ���ݿ�
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
    * ������������prefixspan�㷨������м���Ƶ������
    */
    protected void prefixSpan(String[] prefixSeq,List<List<List<String>>>seqs,int prefixNum){
    	// ���ǰ׺�ó��������еĳ��ȴ���maxFrElemSize,��ֱ������
    	if(prefixNum > this.maxFrElemSize - 1){
    		return;
    	}
    	for(int tTotal = 0;tTotal < total;tTotal++){
    		// ��һ�����a��ͶӰ���ݿ�seqs,ѭ�����������ItemList,���Ƿ����ĳ��item��seqs�ϻ�����Ƶ������eg:<a><b> 
    		int supNum1 = 0;
    		String tempSeq = itemList.get(tTotal);
    		supNum1 = makeout(tempSeq,seqs);

    		if(supNum1 >= minSup){
    			//��ʼ��¼Ƶ������
    			List<String> itemList = new ArrayList<String>();
    			if(prefixNum >= this.minFrElemSize - 1){
    				for(int i=0;i<prefixNum;i++){
    					itemList.add(prefixSeq[i]);
    				}
    				itemList.add(tempSeq);
    				itemList.add(supNum1+"");//���֧����
    				itemList.add((float)supNum1 / seqs.size()+"");//������Ŷ�
    				maxFrSeqs.add(itemList);
    			}
    			List<List<List<String>>> sdSeqs = generateSD(seqs,tempSeq);
    			String prefixSeq2[] = new String[prefixNum+1];
    			for(int e=0;e<prefixNum;e++)
    				prefixSeq2[e] = prefixSeq[e];
    			prefixSeq2[prefixNum] = tempSeq;
    			prefixSpan(prefixSeq2,sdSeqs,prefixNum+1);
    		}
    		//�ڶ������a��ItemList��ĳ������������,���Ƿ���seqs�ǻ����ڴ�����С֧������item eg:<a,b>
    		int supNum2 = 0;
    		String tempSeq1 = prefixSeq[prefixNum-1]+","+itemList.get(tTotal);
    		String tempSeq1s[] = tempSeq1.split(",");
    		supNum2 = makeout(tempSeq1s,seqs);
    		if(supNum2 >= minSup){
    			//��ʼ��¼Ƶ������
    			List<String> itemList = new ArrayList<String>();
    			if(prefixNum >= this.minFrElemSize){
    				for(int i=0;i<prefixNum-1;i++){
    					itemList.add(prefixSeq[i]);
    				}
    				itemList.add(tempSeq1);
    				itemList.add(supNum2+"");//���֧�ֶ�
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
    	log.info("����֧�ֶ�Ϊ{}�����Ϊ{}����item��Ϊ{}",new Integer[]{minSup,minFrElemSize,total});
    	String itemId = null;
    	for(Entry<String,Integer>supEntry:supMap.entrySet()){
    		itemId = supEntry.getKey();
    		//����ͶӰ���ݿ�
    		List<List<List<String>>>sdList = this.generateSD(itemId);
    		String prefixSeq[] = {itemId};
    		this.prefixSpan(prefixSeq,sdList,1);
    		times++;
    		log.info("ִ�е�itemId-{},�Ѿ�ѭ����{}",new String[]{prefixSeq[0],times+""});
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
    * ����ǰ׺����ͶӰ���ݿ�
    * @param seqs �������ݿ� S
    */
    public List<List<List<String>>>generateSD(List<List<List<String>>>seqs,String prefixSeq){
    	return generateSD(seqs,new String[]{prefixSeq});
    }
    /**
    * ����ǰ׺����ͶӰ���ݿ�
    * @param prefixSeq ǰ׺
    */
    public List<List<List<String>>>generateSD(String prefixSeq){
    	return generateSD(sequences,new String[]{prefixSeq});
    }
    /**
    * ����ǰ׺����ͶӰ���ݿ�
    * @param seqs �������ݿ� S
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
    * �����������ж��ַ����е������Ƿ������⼯��(����)�а��� eg:items=[a,b,c] temps=[_,_,c]�Ǵ��ڵ�"_"Ϊ�����ַ���
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
    * �����������ж��ַ����е������Ƿ��ڼ����д���
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
    			n = items.indexOf(temps[0]);//�����ҵ�����ĵ�һ��Ԫ�س����ڼ����е�λ��
    			if(n+length<=size&&n!=-1){// ������Ϲ���,�ҵ�һ������
    				for(int i=1;i<length;i++){ //�ٰ�˳���ҵڶ�������N��
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
//	    	lineInfo = Eng_Similarity.getTokens(lineInfo);  //����ȥ�������ţ�ͣ�ôʵ��ַ���
	    	
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
					String pos = token.get(PartOfSpeechAnnotation.class); //���ʵ�POS
					String lemma = token.get(LemmaAnnotation.class);  //ԭ��
//					text.add(pos);
					if(pos.equals("CD")) //�����ʱ�������  ��230��one TWO������
						text.add(pos);
					else
						text.add(lemma);
				}
			}			
	    	for(String tag: text){
	    		items.add(Arrays.asList(tag));  //����������Ҫ�ĸ�ʽ[word]
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
    	PrefixSpanBuild test = new PrefixSpanBuild(sLists,9,2,10);  //��С֧�ֶȣ�������г��ȣ�����г���
    	test.buildPrefixSpan();
    	System.out.println("�������ݿ����£�");
    	for(List<List<String>>elem:test.sequences){
    		for(List<String>item2s:elem){
    			System.out.print(item2s);
    			System.out.print("      ");
    		}
    		System.out.println();
    	}
    	System.out.println("");
    	System.out.println("");
    	System.out.println("ִ��PrefixSpan�㷨������Ƶ������ģʽ������£�");
    	test.printMaxFrSeq();
    }
    
    public static void main(String[] args) throws IOException{
    	String inputPath = "./10";
    	List<List<List<String>>> sLists = textPos(inputPath);
    	prefixSpan(sLists);	
    }
   }
