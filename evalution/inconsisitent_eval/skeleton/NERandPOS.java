package skeleton;

import java.util.List;
import java.util.Properties;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class NERandPOS {
	public static void main(String[] args) throws Exception {
	    
	    String[] example = {"At least 38 killed in Pakistan by blast outside Lahore park",
	    		"TTP claims Lahore park explosion that killed at least 65"};
	    Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);	
		
	    for (String str : example) {
			
	    	long start = System.currentTimeMillis();
		    String content = "";
		    
		    Annotation document = new Annotation(str);
			pipeline.annotate(document);
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for(CoreMap sentence: sentences) {
				for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
					String ne = token.get(NamedEntityTagAnnotation.class);    // 获取命名实体识别结果  
					content = content + ne+" ";
				}
			}
			long end = System.currentTimeMillis();
			System.out.println(content);
			System.out.println(end-start);
	    }
	}
}
