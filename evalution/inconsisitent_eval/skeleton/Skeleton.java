package skeleton;

/**
 * 增加了word2vec和openie的不�?致检�?
 */
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.gson.Gson;

import edu.knowitall.openie.Argument;
import edu.knowitall.openie.Instance;
import edu.knowitall.openie.OpenIE;
import edu.knowitall.tool.parse.ClearParser;
import edu.knowitall.tool.postag.ClearPostagger;
import edu.knowitall.tool.srl.ClearSrl;
import edu.knowitall.tool.tokenize.ClearTokenizer;
//import SetCoverProblem.Eng_Similarity;
import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import edu.stanford.nlp.naturalli.SentenceFragment;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;
import inconsistent.ie.IEResult;
import inconsistent.ie.JavaOllieWrapper;
import inconsistent.stanford.StanfordPos;
import inconsistent.word2vec.Word2VecHelper;
import net.didion.jwnl.JWNLException;
import preprocess.HybridTokenizer;
import preprocess.Twitter;
import scala.collection.JavaConversions;
import scala.collection.Seq;
import shef.nlp.wordnet.similarity.SimilarityInfo;
import shef.nlp.wordnet.similarity.SimilarityMeasure;


public class Skeleton {
	
	public static void main(String[] args) {
		//调用python运行BOW模型和AP聚类
//		String filename = "2016-sismoecuador.ids";
//		String[] arg = new String[] {"D:\\Software\\Anaconda2\\python", "F:\\llt\\ShowResult\\src\\Skeleton\\BOW.py",filename};
//		try {  
//			System.out.println("start");
//			String a = textPos("Death toll rises to 262 in Ecuador earthquake.");
//			String b = textPos("Ecuador Earthquake: Death Toll Rises To 246    Lets Pray for them!");
//			String[] aa=a.split(" ");
//	        String[] bb=b.split(" ");
//	        
////	        HybridTokenizer hybridTokenizer = new HybridTokenizer();
////		  	List<String> filteredTokenList1 = hybridTokenizer.tokenize(a);
////		  	List<String> filteredTokenList2 = hybridTokenizer.tokenize(b);
////		  	
////		  	String[] aa = filteredTokenList1.toArray(new String[filteredTokenList1.size()]);
////		  	String[] bb = filteredTokenList2.toArray(new String[filteredTokenList2.size()]);
//		  	
//	        
//	        
//		  //计算lcs递归矩阵   
//	        int[][] re = LongestCommonSub.longestCommonSubsequence(aa, bb);  
//	        List<String> str = new ArrayList<String>();
//	        str = LongestCommonSub.print(re, aa, bb, aa.length, bb.length,str);  
//			for(int i=0;i<str.size();i++)
////			Process proc = Runtime.getRuntime().exec(arg);    //filepath是test.py的地�?。可以取相对地址，以项目�?在地�?为根目录  
////			System.out.println(proc.waitFor());  
//	        System.out.println("end");
//	        //PrefixSpanBuild.prefixSpan(PrefixSpanBuild.textPos(filename));
//		} catch (Exception e) { 
//		    e.printStackTrace();  
//		}
		
		//String t = normalizeNum("100,0");
		//System.out.println(t);
//		String line1 = "Live blog: reports 10 members of the Alhaj family have been killed in an attack on Khan Younis -  #GazaUnderAttack";
//		String line2 = "The only remaining members from Al-Batsh family as israeli terror attacks wiped out 18 members #GazaUnderAttack";
		//StanfordPos sp = new StanfordPos();
//		String text = "Reports of series of bombings and two serious injuries in attack on Zaytoun area #GazaUnderAttack #PrayForGaza #EndTheOccupation";
//		//sp = posTagAndOpenIE(text);
//		String text2 = "#IREvCRO. Ireland back at Euros for 1st time in 24 yrs. Last 2 nations returning after 24 yrs FRA (84)  GRE (04)";
//	    long start = System.currentTimeMillis();
//		Properties props = PropertiesUtils.asProperties(
//	            "annotators", "tokenize,ssplit,pos,lemma,ner, parse, dcoref,depparse,natlog,openie",
//	            "openie.splitter.threshold", "0.25",
//	            "openie.max_entailments_per_clause", "1000",
//	            "openie.triple.strict", "false",
//	            "ssplit.isOneSentence", "true",
//	            "tokenize.class", "PTBTokenizer",
//	            "tokenize.language", "en",
//	            "enforceRequirements", "true"
//	    );
//        int i=0;
//        List<String> wordList = new ArrayList<String>();
//        List<String> posList = new ArrayList<String>();
//        List<String> lemmaList = new ArrayList<String>();
//        List<Double> digitList = new ArrayList<Double>(); 
//        List<String> nerList = new ArrayList<String>();
//        List<IEResult> ieResultList = new ArrayList<IEResult>();
//        IEResult ie = new IEResult();
//        String praseTree = null;
//        StanfordPos sp = new StanfordPos();
//	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
//	    Annotation doc = new Annotation(text);
//	    pipeline.annotate(doc);
//	    List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);
//        System.out.println("sentences sizes: " + sentences.size());
//        System.out.println("word\tpos\tlemma\tner\t");//ner");
//	    for (CoreMap sentence : sentences) {
//            // traversing the words in the current sentence
//            // a CoreLabel is a CoreMap with additional token-specific methods
//	       
//           for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
//               // this is the text of the token
//               String word = token.get(TextAnnotation.class);
//               // this is the POS tag of the token
//               String pos = token.get(PartOfSpeechAnnotation.class);
//               // this is the NER label of the token
//               String ner = token.get(NamedEntityTagAnnotation.class);
//               String lemma = token.get(LemmaAnnotation.class);
//               Double digit = 0.0;
//               wordList.add(word);
//               posList.add(pos);
//               lemmaList.add(lemma);
//               nerList.add(ner);
//               digitList.add(digit);
//               System.out.println(word+"\t"+pos+"\t"+lemma+"\t"+ner + "\t");
//           }
//           
//           // Get the OpenIE triples for the sentence
//           Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
//           // Print the triples
//           for (RelationTriple triple : triples) {
//        	   ie.setConf(triple.confidence);
//        	   ie.setArg1(triple.subjectLemmaGloss());
//        	   ie.setRel(triple.relationLemmaGloss());
//        	   ie.setArg2(triple.objectLemmaGloss());
//        	   i++;
//        	   ieResultList.add(ie);
//               System.out.println(triple.confidence + "->\t" +
//                       triple.subjectLemmaGloss() + "->\t" +
//                       triple.relationLemmaGloss() + "->\t" +
//                       triple.objectLemmaGloss());
//           }
//           sp.setIeResult(ieResultList);
//           sp.setWord(wordList);
//           sp.setPos(posList);
//           sp.setLemma(lemmaList);
//           sp.setDigit(digitList);
//           sp.setNer(nerList);
//           sp.setFlag(i);
//           //根据解析树
//           if(sp.getFlag() == 0) {
//        	   
//           }
//           
//	    }
//	    long mid = System.currentTimeMillis();
//	    System.out.println("first sentence cost times: " + (mid-start)/1000.0);
//	    doc = new Annotation(text2);
//	    pipeline.annotate(doc);
//	    sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);
//        System.out.println("sentences sizes: " + sentences.size());
//        System.out.println("word\tpos\tlemma\tner\t");//ner");
//	    for (CoreMap sentence : sentences) {
//            // traversing the words in the current sentence
//            // a CoreLabel is a CoreMap with additional token-specific methods
//	       
//           for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
//               // this is the text of the token
//               String word = token.get(TextAnnotation.class);
//               // this is the POS tag of the token
//               String pos = token.get(PartOfSpeechAnnotation.class);
//               // this is the NER label of the token
//               String ner = token.get(NamedEntityTagAnnotation.class);
//               String lemma = token.get(LemmaAnnotation.class);
//               Double digit = 0.0;
//               wordList.add(word);
//               posList.add(pos);
//               lemmaList.add(lemma);
//               nerList.add(ner);
//               digitList.add(digit);
//               System.out.println(word+"\t"+pos+"\t"+lemma+"\t"+ner + "\t");
//           }
//           
//           // Get the OpenIE triples for the sentence
//           Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
//           // Print the triples
//           for (RelationTriple triple : triples) {
//        	   ie.setConf(triple.confidence);
//        	   ie.setArg1(triple.subjectLemmaGloss());
//        	   ie.setRel(triple.relationLemmaGloss());
//        	   ie.setArg2(triple.objectLemmaGloss());
//        	   i++;
//        	   ieResultList.add(ie);
//               System.out.println(triple.confidence + "->\t" +
//                       triple.subjectLemmaGloss() + "->\t" +
//                       triple.relationLemmaGloss() + "->\t" +
//                       triple.objectLemmaGloss());
//           }
//           System.out.println("to only run e.g., the clause splitter:");
//           // Alternately, to only run e.g., the clause splitter:
//           List<SentenceFragment> clauses = new edu.stanford.nlp.naturalli.OpenIE(props).clausesInSentence(sentence);
//           for (SentenceFragment clause : clauses) {
//             System.out.println(clause.parseTree.toString(SemanticGraph.OutputFormat.LIST));
//           }
//           System.out.println();
//           sp.setIeResult(ieResultList);
//           sp.setWord(wordList);
//           sp.setPos(posList);
//           sp.setLemma(lemmaList);
//           sp.setDigit(digitList);
//           sp.setNer(nerList);
//           sp.setFlag(i);
//           //根据解析树
//           if(sp.getFlag() == 0) {
//        	   
//           }
//	    }
//	    long end =  System.currentTimeMillis();
//	    System.out.println("second sentence cost time: " + (end-mid)/1000.0);
		
        List<IEResult> ieResultList = new ArrayList<IEResult>();
        
        int flag = 0;
        //先使用openie5.0来抽取信息
        OpenIE openIE = new OpenIE(new ClearParser(new ClearPostagger(new ClearTokenizer())), new ClearSrl(), false, false);
        Seq<Instance> extractions = openIE.extract("Boris Johnson accused: 'you're a nasty piece of work'#Brexit #EUref #VoteLeave #LeaveEU #GrassrootsOut #UKIP");
        
        List<Instance> list_extractions = JavaConversions.seqAsJavaList(extractions);
        for(Instance instance : list_extractions) { 
           IEResult ie = new IEResult();
           flag ++;
     	   ie.setConf(instance.confidence());
     	   ie.setArg1(instance.extr().arg1().text());
     	   ie.setRel(instance.extr().rel().text());
     	   List<Argument> list_arg2s = JavaConversions.seqAsJavaList(instance.extr().arg2s());
     	   StringBuilder arg2 = new StringBuilder();
            for(Argument argument : list_arg2s) {
         	   arg2.append(argument.text()).append(" ");
            }
            ie.setArg2(arg2.toString());
            ieResultList.add(ie);
            
            
        }
//        Collections.sort(ieResultList, new Comparator<IEResult>() {
//            @Override
//            public int compare(IEResult o1, IEResult o2) {
//                //降序
//                return o2.getConf().compareTo(o1.getConf());
//            }
//        });
        Collections.sort(ieResultList);
        for(IEResult i : ieResultList) {
        	System.out.println("ie:" + i.getConf() + " " + i.getArg1() + " " + i.getRel() + " " + i.getArg2());
        }
		
		
		
	}
	
	/**
	 * 将一个id，imporant，weight，text转换成只有text的List数组
	 * @param blogList
	 * @return
	 */
	public static List<String> convertBlogToStringList(List<String> blogList){
		Gson gson=new Gson();
		List<String> list = new ArrayList<String>();
		for(String b:blogList) {
			Twitter weibo=gson.fromJson(b, Twitter.class);
			list.add(weibo.getText());
		}
		return list;
	}
	/**
	 * 将一个id，imporant，weight，text转换成只有text的String
	 * @param blogList
	 * @return
	 */
	public static String convertBlogToString(String blog) {
		String b="";
		Gson gson=new Gson();
		Twitter weibo=gson.fromJson(blog, Twitter.class);
		b = weibo.getText();
		return b;
	}
	
	/**
	 * 判断是不是数字，数字包括英文的一些基本数字，和纯数字，还有有逗号的那种数�?
	 * @param s
	 * @return
	 */
	public static boolean isNum(String t) {
		String numberWithCommas = "(\\d+,)+?\\d{3}" + "(?=([^,]|$))";
		String timeLike   = "\\d+:\\d+";
		String numNum     = "\\d+\\.\\d+";
		String isNum = "[0-9]+";
		String []englishNum = {"zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen",
				"fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
				"eighty","ninety","hundred","thousand","million","billion"};
		if(t.matches(numberWithCommas) || t.matches(numNum) || t.matches(timeLike) || t.matches(isNum)) {
			t.replaceAll("[.,]", "");
			System.out.println("进来了");
			return true;
		}
		for(int i=0;i<englishNum.length;i++) {
			if(t.equals(englishNum[i])) {
				return true;
			}
		
		}
		return false;
	}
	
	/**
	 * 数字归一化,出现英文的，和符号把他换成数字
	 */
	public static String normalizeNum(String t) {
		String []englishNum = {"zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen",
				"fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
				"eighty","ninety","hundred","thousand","million","billion"};
		for(int i=0;i<englishNum.length;i++) {
			if((t.toLowerCase()).equals(englishNum[i])) {
				if(i>=0 && i<= 20) {
					t = String.valueOf(i);
				}
				if(21 == i)
					t = "30";
				if(22 == i)
					t = "40";
				if(23 == i)
					t = "50";
				if(24 == i)
					t = "60";
				if(25 == i)
					t = "70";
				if(26 == i)
					t = "80";
				if(27 == i)
					t = "90";
				if(28 == i)
					t = "100";
				if(29 == i)
					t = "1000";
				if(30 == i)
					t = "1000000";
				if(31 == i)
					t = "1000000000";
				return t;
			}
		}
		t = t.replaceAll("[_`~[-]!@#$%^&*()+=|{}':;',a-zA-z\\[\\]\\\\<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。，、？]|\n|\r|\t", "");
		
		//极端情况，处理完只剩空字符了，说明这个没用
		if(t.isEmpty() || t.equals("") || t.equals(" ")) {
			t = "-1";
		}
		
		return t;
	}





	private Object triple;
	
	
	
	/**
	 * stanford词性标注 and openie
	 * @param text
	 * @return
	 */
	public static StanfordPos posTagAndOpenIE(String text,StanfordCoreNLP pipeline, OpenIE openIE) {
		

        List<String> wordList = new ArrayList<String>();
        List<String> posList = new ArrayList<String>();
        List<String> lemmaList = new ArrayList<String>();
        List<Double> digitList = new ArrayList<Double>(); 
        List<String> nerList = new ArrayList<String>();
        List<IEResult> ieResultList = new ArrayList<IEResult>();
        
        String praseTree = null;
        StanfordPos sp = new StanfordPos();
	    Annotation doc = new Annotation(text);
        int flag = 0;
        //先使用openie5.0来抽取信息
        System.out.println("出错前抽取的text->" + text);
        Seq<Instance> extractions = openIE.extract(text);
        
        List<Instance> list_extractions = JavaConversions.seqAsJavaList(extractions);
        for(Instance instance : list_extractions) { 
        	IEResult ie = new IEResult();
		    flag ++;
		    ie.setConf(instance.confidence());
		    ie.setArg1(instance.extr().arg1().text());
		    ie.setRel(instance.extr().rel().text());
		    List<Argument> list_arg2s = JavaConversions.seqAsJavaList(instance.extr().arg2s());
		    StringBuilder arg2 = new StringBuilder();
            for(Argument argument : list_arg2s) {
         	   arg2.append(argument.text()).append(" ");
            }
            ie.setArg2(arg2.toString());
            ieResultList.add(ie);

//        	StringBuilder sb = new StringBuilder();
//        	
//            sb.append(instance.confidence())
//                .append("\t>")
//                .append(instance.extr().arg1().text())
//                .append("\t>")
//                .append(instance.extr().rel().text())
//                .append("\t>")
//                .append(arg2);
//                
//            
//            System.out.println(sb.toString());
            
            
        }
	    
	    
	    pipeline.annotate(doc);
	    List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);
        //System.out.println("sentences sizes: " + sentences.size());
        //System.out.println("word\tpos\tlemma\tner\t");//ner");
	    for (CoreMap sentence : sentences) {
            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
	       
           for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
               // this is the text of the token
               String word = token.get(TextAnnotation.class);
               // this is the POS tag of the token
               String pos = token.get(PartOfSpeechAnnotation.class);
               // this is the NER label of the token
               String ner = token.get(NamedEntityTagAnnotation.class);
               String lemma = token.get(LemmaAnnotation.class);
               Double digit = 0.0;
               wordList.add(word);
               posList.add(pos);
               lemmaList.add(lemma);
               nerList.add(ner);
               digitList.add(digit);
               //System.out.println(word+"\t"+pos+"\t"+lemma+"\t"+ner + "\t");
           }

           if(flag == 0) {  //openIe没取出东西 ，调用他的子包stanford
        	   System.out.println("openie没提起出东西，调用stanford ie");
               // Get the OpenIE triples for the sentence
               Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
               // Print the triples
               for (RelationTriple triple : triples) {
            	   IEResult ie = new IEResult();
            	   ie.setConf(triple.confidence);
            	   ie.setArg1(triple.subjectLemmaGloss());
            	   ie.setRel(triple.relationLemmaGloss());
            	   ie.setArg2(triple.objectLemmaGloss());
            	   flag++;
            	   ieResultList.add(ie);
                   //System.out.println(triple.confidence + "->\t" +
                           //triple.subjectLemmaGloss() + "->\t" +
                           //triple.relationLemmaGloss() + "->\t" +
                           //triple.objectLemmaGloss());
               }
               
           }
           else {
        	   System.out.println("openie成功抽取");
           }
           Collections.sort(ieResultList);  //按conf降序
           sp.setIeResult(ieResultList);
           sp.setWord(wordList);
           sp.setPos(posList);
           sp.setLemma(lemmaList);
           sp.setDigit(digitList);
           sp.setNer(nerList);
           sp.setFlag(flag);


           //根据解析树
           if(sp.getFlag() == 0) {
        	   System.out.println("stanford ie没提取出东西，调用解析树");
           }
           
	    }
	    return sp;
	}
	
	
	/**
	 * 判断参数中是否存在数字
	 * @param arg
	 * @return
	 */
	public static int isExistNumInArg(StanfordPos sp, String[] arg)
	{
		for(String s:arg) {
			for(int i=0;i<sp.getWord().size();i++) {
				if(s.equals(sp.getWord().get(i)) && sp.getPos().get(i).equals("CD")){
					return i;
				}
			}
		}
		return -1;
		
	}
	
	/**
	 * 找到参数的原型
	 * @param sp
	 * @param arg
	 * @return
	 */
	public static String findLemma(StanfordPos sp,String arg) {
		StringBuffer sb = new StringBuffer();
		String[] args = arg.split(" ");
		int flag=0;
		for(String t:args) {
			for(int i=0;i<sp.getWord().size();i++) {
				if(t.equals(sp.getWord().get(i))) {
					sb.append(sp.getLemma().get(i) + " ");
					flag++;
				}
				if(flag == args.length)
					return sb.toString();
				
			}
			
		}
		return sb.toString();
	}
	
	
	/**
	 * 两句话的相似度超过0.8，调用这个不一致检测算法。里面包括了stanford，ollie，wordnet
	 * @param line1
	 * @param line2
	 * @param ollieWrapper
	 * @param sim
	 * @return
	 * @throws JWNLException
	 * @throws IOException 
	 */
	public static boolean isInconsistent(String line1,String line2,StanfordCoreNLP pipeline, SimilarityMeasure sim,OpenIE openIE) throws JWNLException, IOException {
		String fileName = "D:\\chao\\Data\\cantextraction\\";
		File file=new File(fileName);
		if(!file.exists()) file.mkdirs();
		fileName = fileName + "list.txt";
		System.out.println("创建文件夹了");
		BufferedWriter bw=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName,true),"utf-8"));
		StanfordPos sp1 = new StanfordPos();
    	StanfordPos sp2 = new StanfordPos();
		//stanford 进行词性标注
		sp1 = posTagAndOpenIE(line1,pipeline,openIE);
		sp2 = posTagAndOpenIE(line2,pipeline,openIE);
		//判断是否出现CD 即中文的数字，把他归一化
		List<String> wordList1 = sp1.getWord();
		List<String> wordList2 = sp2.getWord();
		List<Double> digitList1 = sp1.getDigit();
		List<Double> digitList2 = sp2.getDigit();
			for(int i=0;i<sp1.getPos().size();i++) {  
			if("CD".equals(sp1.getPos().get(i).toUpperCase()) && !"|".equals(sp1.getLemma().get(i)) && "NUMBER".equals(sp1.getNer().get(i).toUpperCase())) {  //标注是数字  因为没有ner和decrof会识别错|为数字CD
				String word = sp1.getWord().get(i);
				if(!word.isEmpty() || !word.equals("") || !word.equals(" ")) {
					word = normalizeNum(word);
					System.out.println("归一化数字1："+word);
					digitList1.set(i, Double.parseDouble(word)); //更新归一化后的数字
				}

			}
		}
		sp1.setDigit(digitList1);  //更新这个sp1里的digitList；
		
		for(int i=0;i<sp2.getPos().size();i++) {
			if("CD".equals(sp2.getPos().get(i).toUpperCase())  && !"|".equals(sp2.getLemma().get(i)) && "NUMBER".equals(sp2.getNer().get(i).toUpperCase())) {  //标注是数字 因为没有ner和decrof会识别错|为数字CD
				String word = sp2.getWord().get(i);
				if(!word.isEmpty() || !word.equals(" ") || !word.equals("")) {
					word = normalizeNum(word);
					System.out.println("归一化数字1：" + word);
					digitList2.set(i, Double.parseDouble(word)); //更新归一化后的数字
				}
			}
		}
		sp2.setDigit(digitList2);  //更新这个sp1里的digitList;
		
		//IE的抽取
		List<IEResult> IEList1 = sp1.getIeResult();
		List<IEResult> IEList2 = sp2.getIeResult();


        if(sp1.getFlag()==0) {
        	bw.write(line1 + "\n");
        	System.out.println("line1 not extract");
        }
        	
        if(sp2.getFlag()==0) {
        	bw.write(line2 + "\n");
        	System.out.println("line2 not extract");
        }
        	
        bw.flush();
        bw.close();
        
        if(sp1.getFlag() == 0 || sp2.getFlag() == 0) {
        	return false;
        }
        //两个句子都抽出来了，一句话可能有不同的抽取结果
        IEResult ie1 = new IEResult();
        IEResult ie2 = new IEResult();
        List<String> posList1 = sp1.getPos();
        List<String> posList2 = sp2.getPos();
        List<String> lemmaList1 = sp1.getLemma();
        List<String> lemmaList2 = sp2.getLemma();
        for(int i=0;i<IEList1.size();i++) {   //遍历多个提取结果行
    		ie1 = IEList1.get(i);
    		String rel1 = ie1.getRel();
    		
    		//判断这个rel要提取多少个词
    		String[] rel1List = rel1.split(" ");
    		int rel_flag = 0;  //判断是否有动词 ，没动词就全要了
    		if(rel1List.length != 1) {   //如果rel中不止一个词
    			for(int m=0;m<rel1List.length;m++) {
    				String r = rel1List[m];
    				
    				for(int j = 0; j < wordList1.size(); j++) {
    					if(r.equals(wordList1.get(j)) && posList1.get(j).startsWith("V"))  //动词
    					{
    						rel_flag ++;
    						rel1 = lemmaList1.get(j);
    						if((m+1 < rel1List.length) && rel1List[m+1].equals(wordList1.get(j+1)) && (posList1.get(j+1).equals("IN") || posList1.get(j+1).equals("RP"))) {
    							rel1 += " " + lemmaList1.get(j+1);    //动词后面有跟介词的
    						}
    						
    					}
    				}

    			}
				if(rel_flag == 0) {   //没有动词存在
    				rel1 = "";
    				for(int m=0;m<rel1List.length;m++) {
    					String r = rel1List[m];
    					for(int j=0;j<wordList1.size();j++) {
    						if((r.trim()).equals(wordList1.get(j))) {
    							rel1 += " " + lemmaList1.get(j);
    						}
    					}
    				}
    				
				}
				System.out.println("没有replace之前的rel1："+rel1);
				rel1 = rel1.replaceAll("[_`~[-]!@#$%^&*()+=|{}':;',\\[\\]\\\\<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。，、？]|\n|\r|\t", "").toLowerCase();
    			System.out.println("replace后的rel1："+rel1);
    		}
    		else {
    			for(int j=0; j <wordList1.size(); j++) {
    				if((rel1.trim()).equals(wordList1.get(j)))
    					rel1 = lemmaList1.get(j);
    			}
    			rel1 = rel1.replaceAll("[_`~[-]!@#$%^&*()+=|{}':;',\\[\\]\\\\<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。，、？]|\n|\r|\t", "").toLowerCase();

    		}
			if(rel1.isEmpty() || rel1.equals("") || rel1.equals(" ")) {
				continue;
			}
    		
        	for(int t=0;t<IEList2.size();t++) {
        		ie2 = IEList2.get(t);
        		String rel2 = ie2.getRel();
        		//先判断rel的相似性
        		//判断这个rel要提取多少个词
        		String[] rel2List = rel2.split(" ");
        		int rel2_flag = 0;  //判断是否有动词 ，没动词就全要了
        		if(rel2List.length != 1) {   //如果rel中不止一个词
        			for(int m=0;m<rel2List.length;m++) {
        				String r = rel2List[m];
        				for(int j = 0; j < wordList2.size(); j++) {
        					if((r.trim()).equals(wordList2.get(j)) && posList2.get(j).startsWith("V"))  //动词
        					{
        						rel2_flag ++;
        						rel2 = lemmaList2.get(j);
        						if((m+1 < rel2List.length) && rel2List[m+1].equals(wordList2.get(j+1)) && (posList2.get(j+1).equals("IN") || posList2.get(j+1).equals("RP") )) {
        							rel2 += " " + lemmaList2.get(j+1);    //动词后面有跟介词的
        						}
        						
        					}
        				}
        			}
        			if(rel2_flag == 0) {
        				rel2 = "";
        				for(int m=0;m<rel2List.length;m++) {
        					String r = rel2List[m];
        					for(int j=0;j<wordList2.size();j++) {
        						if((r.trim()).equals(wordList2.get(j))) {
        							rel2 += " "+lemmaList2.get(j);
        						}
        					}
        				}
        				
        			}
        			rel2 = rel2.replaceAll("[_`~[-]!@#$%^&*()+=|{}':;',\\[\\]\\\\<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。，、？]|\n|\r|\t", "").toLowerCase();
        			
        			
        		}
        		else {
        			for(int j=0; j <wordList2.size(); j++) {
        				if((rel2.trim()).equals(wordList2.get(j)))
        					rel2 = lemmaList2.get(j);
        			}
        			rel2 = rel2.replaceAll("[_`~[-]!@#$%^&*()+=|{}':;',\\[\\]\\\\<>/?~！@#￥%……&*（）——+|{}【】‘；：”“’。，、？]|\n|\r|\t", "").toLowerCase();
        		}
        		if(rel2.isEmpty() || rel2.equals("") || rel2.equals(" ")) {
        			continue;
        		}
        		System.out.println("出错的rel1:" +rel1  + "和rel2:" + rel2);
        		//开始判断rel1和rel2，使用wordnet
        		Double s = 0.0;
        		SimilarityInfo si = sim.getSimilarity(rel1, rel2);
        		if(si != null)
        			s = si.getSimilarity();
        		if(s > 0.14)  //网上看的说0.14算是比较高的相似度
        		{
        			String arg11 = ie1.getArg1();
        			String arg12 = ie1.getArg2();
        			String arg21 = ie2.getArg1();
        			String arg22 = ie2.getArg2();
        			//找出他们的原型
        			arg11 = findLemma(sp1, arg11);
        			arg12 = findLemma(sp1, arg12);
        			arg21 = findLemma(sp2, arg21);
        			arg22 = findLemma(sp2, arg22);
        			
        			
        			//调用LCS 比较参数
        			String[] aa=arg11.split(" ");    //每一条微博里用空格分隔
        	        String[] ba=arg21.split(" ");
        	        String[] ab=arg12.split(" ");
        	        String[] bb=arg22.split(" ");
        	        //判断是否有数字
        	        int flag1 = isExistNumInArg(sp1,aa);
        	        int flag2 = isExistNumInArg(sp1,ba);
        	        int flag3 = isExistNumInArg(sp2,ab);
        	        int flag4 = isExistNumInArg(sp2,bb);
        	        
        	        int flag = 0;
        	        //计算lcs递归矩阵  
        	        int[][] re = LongestCommonSub.longestCommonSubsequence(aa, ba);  
        	        List<String> str = new ArrayList<String>();
        	        str = LongestCommonSub.print(re, aa, ba, aa.length, ba.length,str);  
        	        if(str.size()>=Math.min(aa.length, ba.length)/2) { //如果公共最长序列大于两句话中最短的一半，就是相似的 冲突了
        	        	if(flag1 != -1 && flag2 != -1) {
        	        		if(sp1.getDigit().get(flag1) != sp2.getDigit().get(flag2)) {
        	        			flag ++;
        	        		}
        	        	}
        	        	else if(flag1 != -1 || flag2 != -1) {
        	        		
        	        	}
        	        	else {
        	        		flag++;
        	        	}
        	        	
        	        	
        	        }
        	        if(flag == 1) {
        	        	re = LongestCommonSub.longestCommonSubsequence(ab, bb);
        	        	str = new ArrayList<String>();
	        	        str = LongestCommonSub.print(re, ab, bb, ab.length, bb.length,str);  
		        	    if(str.size()>=Math.min(ab.length, bb.length)/2) { //如果公共最长序列大于两句话中最短的一半，就是相似的 冲突了
	        	        	if(flag3 != -1 && flag4 != -1) {
	        	        		if(sp1.getDigit().get(flag3) != sp2.getDigit().get(flag4)) {
	        	        			flag ++;
	        	        		}
	        	        	}
	        	        	else if(flag3 != -1 || flag4 != -1) {
	        	        		
	        	        	}
	        	        	else {
	        	        		flag++;
	        	        	}
	        	        }
		        	    /**else {
	        				return false;
	        			}**/
        	        }
        	        if(flag == 2)
        	        	return true;
	        	    if(flag == 0) {
	        	        //计算lcs递归矩阵  
	        	        re = LongestCommonSub.longestCommonSubsequence(aa, bb);  
	        	        str = new ArrayList<String>();
	        	        str = LongestCommonSub.print(re, aa, bb, aa.length, bb.length,str);  
	        	        if(str.size()>=Math.min(aa.length, bb.length)/2) { //如果公共最长序列大于两句话中最短的一半，就是相似的 冲突了
	        	        	if(flag1 != -1 && flag4 != -1) {
	        	        		if(sp1.getDigit().get(flag1) != sp2.getDigit().get(flag4)) {
	        	        			flag ++;
	        	        		}
	        	        	}
	        	        	else if(flag1 != -1 || flag4 != -1) {
	        	        		
	        	        	}
	        	        	else {
	        	        		flag++;
	        	        	}
	        	        }
	        	        if(flag == 1) {
		        	        //计算lcs递归矩阵  
		        	        re = LongestCommonSub.longestCommonSubsequence(ab, ba);  
		        	        str = new ArrayList<String>();
		        	        str = LongestCommonSub.print(re, ab, ba, ab.length, ba.length,str);  
		        	        if(str.size()>=Math.min(ab.length, ba.length)/2) { //如果公共最长序列大于两句话中最短的一半，就是相似的 冲突了
		        	        	if(flag2 != -1 && flag3 != -1) {
		        	        		if(sp1.getDigit().get(flag2) != sp2.getDigit().get(flag3)) {
		        	        			flag ++;
		        	        		}
		        	        	}
		        	        	else if(flag2 != -1 || flag3 != -1) {
		        	        		
		        	        	}
		        	        	else {
		        	        		flag++;
		        	        	}
		        	        }
		        	        if(flag == 2) {
		        	        	return true;
		        	        }
		        	        /**else {
		        			return false;
		        			}**/
	        	        }/**else {
	        				return false;
	            		}**/
	        	    }
	        	    
        			
        		}/**else {
        			return false;
        		}**/
        		
        	}
        }
        return false;
	}
	
	
	/**
	 * 调用word2vec的工具包进行blog1里的冲突检测，用语义相似度，低于0.8的算不冲突， 2018.12.18 by chao
	 * @param line1
	 * @param line2
	 * @return
	 * @throws IOException
	 * @throws JWNLException 
	 */
	public static boolean isConflictBasicBlog(Word2VecHelper w2h, StanfordCoreNLP pipeline, SimilarityMeasure sim, OpenIE openIE, String line1, String line2, String tokenLine1, String tokenLine2) throws IOException, JWNLException {
		//System.out.println("jinlai");
    	double score = w2h.sentenceSimilairy2(line1,line2); //比较两句话之间的相似度
    	line1 = convertBlogToString(line1);
    	line2 = convertBlogToString(line2);
		if(score >= 0.6) {
			System.out.println("不一致检测中大于0.6调用了extraction");
			boolean flag = isInconsistent(line1,line2,pipeline,sim,openIE);
	        return flag;
		}
		return false;
	}
	
	/**
	 * 后面的一条微博和原先的旧report全部做相似性比较,判断是否可以剪枝
	 * @param oldbloglist
	 * @return
	 * @throws IOException
	 */
	public static boolean prune(Word2VecHelper w2h,List<String> oldbloglist,String line) throws IOException {
		int flag = 0;
		oldbloglist = convertBlogToStringList(oldbloglist);
		line = convertBlogToString(line);
		for(String oldblog:oldbloglist) {
			double score = w2h.sentenceSimilairy2(oldblog,line); //比较两句话之间的相似度
			if(score > 0.6) {
				flag++;
			}
		}
		if(flag == 0)
			return true;  //可以剪枝
		else
			return false; //还需要进行下一步判断
	}
	
	/**
	 * 根据得分，返回他们原数组的位置 （没用到）
	 * @param scores
	 * @param index
	 * @return
	 */
	public static int[] sort_score(List<Double> scores) {
		int[] index = new int[scores.size()];
	    for (int i = 0; i < scores.size(); i++)
	        index[i] = i;
	    for (int i = 0; i < scores.size() - 1; i++)
	    {
	        for (int j = i + 1; j < scores.size(); j++)
	        {
	            //if (src[j] < src[i])
	            if ( scores.get(index[j]) > scores.get(index[i]) )
	            {
	                int tmp = index[i];
	                index[i] = index[j];
	                index[j] = tmp;
	            }             
	        }
	    }
	    return index;
	}
	

	
	
	
/**
	public static boolean isConflict(String line1,String line2) throws IOException {
		
//		HybridTokenizer hybridTokenizer = new HybridTokenizer();
//	  	List<String> filteredTokenList1 = hybridTokenizer.tokenize(line1);
//	  	List<String> filteredTokenList2 = hybridTokenizer.tokenize(line2);
//	  	String[] aa = new String[filteredTokenList1.size()];
//	  	filteredTokenList1.toArray(aa); 
//	  	String[] bb = new String[filteredTokenList2.size()];
//	  	filteredTokenList2.toArray(bb); 
	  	
		String[] aa=line1.split(" ");    //每一条微博里用空格分�?
        String[] bb=line2.split(" ");
	  
        //计算lcs递归矩阵  
        int[][] re = LongestCommonSub.longestCommonSubsequence(aa, bb);  
        List<String> str = new ArrayList<String>();
        str = LongestCommonSub.print(re, aa, bb, aa.length, bb.length,str);  
        if(str.size()>=Math.min(aa.length, bb.length)/2) { //如果�?长公共子序列长度大于较短的句子长度的�?�?
        	System.out.println("LongestCommonSub: "+str);
//        	System.out.println("conflict!!!!!!");
//        	System.out.println(line1);
//        	System.out.println(line2);
        	return true;
        }
        return false;
	}
	**/
	public static boolean isConflict2(String line1,String line2) throws IOException {
		
//		HybridTokenizer hybridTokenizer = new HybridTokenizer();
//	  	List<String> filteredTokenList1 = hybridTokenizer.tokenize(line1);
//	  	List<String> filteredTokenList2 = hybridTokenizer.tokenize(line2);
//	  	String[] aa = new String[filteredTokenList1.size()];
//	  	filteredTokenList1.toArray(aa); 
//	  	String[] bb = new String[filteredTokenList2.size()];
//	  	filteredTokenList2.toArray(bb); 
	  	
		String[] aa=line1.split(" ");    //每一条微博里用空格分�?
        String[] bb=line2.split(" ");
	  
        //计算lcs递归矩阵  
        int[][] re = LongestCommonSub.longestCommonSubsequence(aa, bb);  
        List<String> str = new ArrayList<String>();
        str = LongestCommonSub.print(re, aa, bb, aa.length, bb.length,str);  
        if(str.size()>=Math.min(aa.length, bb.length)/2) { //如果�?长公共子序列长度大于较短的句子长度的�?�?
        	System.out.println("LongestCommonSub: "+str);
//        	System.out.println("conflict!!!!!!");
//        	System.out.println(line1);
//        	System.out.println(line2);
        	return true;
        }
        return false;
	}
	
}
