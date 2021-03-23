import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import com.google.gson.Gson;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import preprocess.HybridTokenizer;
import preprocess.Twitter;
import skeleton.LongestCommonSub;

public class ResultInconsistentRate_NN {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String inputDir = "E:\\chao\\nnsummary_our\\unified-summarization-master\\data\\twitter_final_all\\decode_test_800maxenc_4beam_35mindec_120maxdec_ckpt-8000_beam\\decoded\\";
		String outputDir = "D:\\chao\\nnsummary_our\\inconsistent\\EA\\";
		getInconsistentRate(inputDir,outputDir);
		
	}
	
	public static void getInconsistentRate(String inputDir, String outputDir) throws Exception{
		double avg = 0.0;
		File dir = new File(inputDir);
		if(!dir.isDirectory()){
			System.out.println("该路径下不是文件夹！");
		}else {
			File f=new File(outputDir);
			if(!f.exists()) 
				f.mkdirs();
			String outputFile = outputDir + "inconsistent_rate_IAEA_9m28_abrandom.txt";
			File[] fileList = dir.listFiles();
			HybridTokenizer hybridTokenizer = new HybridTokenizer();
			for(File file : fileList){
				String filename=file.getName();
				if(filename.equals("goodSummary")) {
					continue;
				}
				String input = inputDir+"\\"+filename;
				BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(input),"utf-8"));
				BufferedWriter bw=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile,true),"utf-8"));
    			String lineInfo=" ";
    			List<String> blogList = new ArrayList<String>();
    			while((lineInfo=br.readLine())!=null){
    				blogList.add(lineInfo.trim());
    			}
    			System.out.println(blogList.size());
    			int number = 0;
				String[] aa = null;
				String[] bb = null;
				int[][] re = null;
				List<String> str = new ArrayList<String>();
				int sum = 0;
    			for(int i=0;i<blogList.size();i++) {
    				String s1 = blogList.get(i);
    				List<String> filteredTokenList1 = hybridTokenizer.tokenize(s1);
    				for(int j=0;j<blogList.size();j++) {
    					if(i == j) {
    						continue;
    					}
    					String s2 = blogList.get(j);
    					List<String> strs = new ArrayList<String>();
    					strs.add(s1);
    					strs.add(s2);
    					List<Map<String,String>> result = new ArrayList<Map<String,String>>();
    					result = ner(strs);

    					List<String> filteredTokenList2 = hybridTokenizer.tokenize(s2);

    					Map<String,String> map1 = result.get(0);
    					Map<String,String> map2 = result.get(1);
    					int flag11=0,flag21=0,flag12=0,flag22=0;
    					String num1 = "";
    					String num2 = "";
    					// all have number
    					if(map1.get("number") != null && !map1.get("number").equals("")) {
    						flag11++;
    						String num = map1.get("number");
    						for(int m=0;m<filteredTokenList1.size();m++) {
    							if(filteredTokenList1.get(m).equals(num.trim())) {
    								filteredTokenList1.remove(m);  //到时候直接加在最后面
    								num1 = normalizeNum(num.trim());
    							}
    							
    						}
    						
    					}
    					
    					if(map2.get("number") != null && !map2.get("number").equals("")) {
    						flag21++;
    						String num = map2.get("number");
    						for(int m=0;m<filteredTokenList2.size();m++) {
    							if(filteredTokenList2.get(m).equals(num.trim())) {
    								filteredTokenList2.remove(m);   //到时候直接加在最后面
    								num2 = normalizeNum(num.trim());
    							}
    						}
    						
    					}
    					String location1 = "1";
    					String location2 = "2";
    					if(map1.get("location") != null && !map1.get("location").equals("")) {
    						flag12++;
    					}
    					
    					if(map2.get("location") != null && !map2.get("location").equals("")) {
    						flag22++;
    					}
    					aa = new String[filteredTokenList1.size()];
    					filteredTokenList1.toArray(aa);
    					bb = new String[filteredTokenList2.size()];
    					filteredTokenList2.toArray(bb);
    					
    					re = LongestCommonSub.longestCommonSubsequence(aa, bb);
    	    	        str = LongestCommonSub.print(re, aa, bb, aa.length, bb.length,str);
    	    	        if(flag11 != 0 && flag21 != 0) {
    	    	        	str.add("numeral");
    	    	        }
    	    	        if(flag12 != 0 && flag22 != 0 ) {
    	    	        	str.add("place");
    	    	        }
    	    	        String[] loc1 = {""};
    	    	        String[] loc2 = {""};
    	    	        int flag = 0;
    	    	        if(flag12 != 0 && flag22 != 0) {
    	    	        	location1 = map1.get("location");
    	    	        	location2 = map2.get("location");
    	    	        	loc1 = location1.split("ouyang ");
    	    	        	loc2 = location2.split("ouyang ");
    	    	        	for(int l=0;l<loc1.length;l++) {
    	    	        		for(int m=0;m<loc2.length;m++) {
    	    	        			if(loc1[l].equals(loc2[m])) {
    	    	        				flag++;
    	    	        			}
    	    	        		}
    	    	        	}
    	    	        	
    	    	        }
    	    	        if(str.size()>=Math.max(aa.length, bb.length)*0.5) {
    	    	        	if(flag == 0) {   //地点都不一样，或者说没有地点
    	    	        		number++;
    	    	        		break;
    	    	        	}
    	    	        	
    	    	        	else{      //地点一样，数量一样
    	    	        		if(num1.equals(num2)) {

    	    	        		}else {   //地点一样，数量不一样
    	    	        			number++;
    	    	        			break;
    	    	        		}
    	    	        	}
    	    	        }
    	    	        str.clear();
    	    	        sum++;
    				}
    			}
    			if(sum == 0) {  //only one blog
    				sum = 1;
    			}
    			double rate = (number*1.0)/blogList.size();
    			BigDecimal   b   =   new   BigDecimal(rate);  
    			double   f1   =   b.setScale(3,   BigDecimal.ROUND_HALF_UP).doubleValue();
    			avg += f1;
    			System.out.println(f1);
    			bw.write(f1 + "\n");
    			bw.flush();
    			bw.close();
    			br.close();
			}
		}
		System.out.println("avg= " + avg/25);
	}
	
	
	public static List<Map<String,String>> ner(List<String> strs) {
		List<Map<String,String>> result = new ArrayList<Map<String,String>>();
	    Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);	
		
	    for (String str : strs) {
			
	    	long start = System.currentTimeMillis();
		    String content = "";
		    Map<String,String> map = new java.util.HashMap<String,String>();
		    Annotation document = new Annotation(str);
			pipeline.annotate(document);
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for(CoreMap sentence: sentences) {
				for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
					String ne = token.get(NamedEntityTagAnnotation.class);    // 获取命名实体识别结果
					if("NUMBER".equals(ne.toUpperCase()))
						map.put("number", token.toString().split("-")[0]);
					if(("LOCATION".equals(ne.toUpperCase())) || ("COUNTRY".equals(ne.toUpperCase())|| ("CITY".equals(ne.toUpperCase())))) {
						if(map.get("location")!=null && !map.get("location").equals("")) {
							content = map.get("location");
							map.put("location", content + "ouyang " + token.toString().split("-")[0]);
						}
						else {
							map.put("location", token.toString().split("-")[0]);
						}
					}
						
//					if("CITY".equals(ne.toUpperCase())) {
//						if(map.get("city")!=null && !map.get("city").equals("")) {
//							content = map.get("city");
//							map.put("city", content + "ouyang " + token.toString().split("-")[0]);
//						}
//						else {
//							map.put("city", token.toString().split("-")[0]);
//						}
//					}
				}
				result.add(map);
			}
//			long end = System.currentTimeMillis();
//			System.out.println(content);
//			System.out.println(end-start);
	    }
		
		
		return result;
	}
	
//	/**
//	 * 数字归一化,出现英文的，和符号把他换成数字
//	 */
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

}
