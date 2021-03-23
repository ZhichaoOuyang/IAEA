/**
 * 
 */
package skeleton;

import java.util.List;

import scala.collection.JavaConversions;
import scala.collection.Seq;
import edu.knowitall.openie.Argument;
import edu.knowitall.openie.Instance;
import edu.knowitall.openie.OpenIE;
import edu.knowitall.tool.parse.ClearParser;
import edu.knowitall.tool.postag.ClearPostagger;
import edu.knowitall.tool.srl.ClearSrl;
import edu.knowitall.tool.tokenize.ClearTokenizer;

/**
 * @author harinder
 *
 */
public class RunMe {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("---Started---");
		long start = System.currentTimeMillis();
		OpenIE openIE = new OpenIE(new ClearParser(new ClearPostagger(new ClearTokenizer())), new ClearSrl(), false, false);
		long start2 = System.currentTimeMillis();
		System.out.println("load cost time: " + (start2 -start)/1000.0);
        Seq<Instance> extractions = null;
        extractions = openIE.extract("Jack and Jill visited India, Japan and South Korea.");
        
        List<Instance> list_extractions = JavaConversions.seqAsJavaList(extractions);
        System.out.println(list_extractions.size());
        for(Instance instance : list_extractions) { 
        	StringBuilder sb = new StringBuilder();
        	
            sb.append(instance.confidence())
                .append("\t>")
                .append(instance.extr().context())
                .append("\t>")
                .append(instance.extr().arg1().text())
                .append("\t>")
                .append(instance.extr().rel().text())
                .append("\t>");
            
            List<Argument> list_arg2s = JavaConversions.seqAsJavaList(instance.extr().arg2s());
            System.out.println(list_arg2s.size());
            for(Argument argument : list_arg2s) {
            	sb.append(argument.text()).append("; ");
            }
            
            System.out.println(sb.toString());
        }
        long end = System.currentTimeMillis();
        System.out.println("first cost time: "+" " +  (end - start2)/1000.0);
        
        extractions = openIE.extract("I can't believe there were no warning signs this pilot had a troubled mind prior to this flight #Germanwings :( #depression? #terrorism?");
        
        list_extractions = JavaConversions.seqAsJavaList(extractions);
        System.out.println(list_extractions.size());
        for(Instance instance : list_extractions) { 
        	StringBuilder sb = new StringBuilder();
        	
            sb.append(instance.confidence())
                .append("\t>")
                .append(instance.extr().context())
                .append("\t>")
                .append(instance.extr().arg1().text())
                .append("\t>")
                .append(instance.extr().rel().text())
                .append("\t>");
            
            List<Argument> list_arg2s = JavaConversions.seqAsJavaList(instance.extr().arg2s());
            System.out.println(list_arg2s.size());
            for(Argument argument : list_arg2s) {
            	sb.append(argument.text()).append("; ");
            }
            
            System.out.println(sb.toString());
        }
        
        long end2 = System.currentTimeMillis();
        System.out.println("second cost time: " + (end2 - end)/1000.0);
        
        
        extractions = openIE.extract("Excellent decision. All 7 Gorkha Regiments of Indian Army to send Nepali Gorkha jawans to Nepal with medical officers to assist operations.");
        
        list_extractions = JavaConversions.seqAsJavaList(extractions);
        System.out.println(list_extractions.size());
        for(Instance instance : list_extractions) { 
        	StringBuilder sb = new StringBuilder();
        	
            sb.append(instance.confidence())
                .append("\t>")
                .append(instance.extr().context())
                .append("\t>")
                .append(instance.extr().arg1().text())
                .append("\t>")
                .append(instance.extr().rel().text())
                .append("\t>");
            
            List<Argument> list_arg2s = JavaConversions.seqAsJavaList(instance.extr().arg2s());
            System.out.println(list_arg2s.size());
            for(Argument argument : list_arg2s) {
            	sb.append(argument.text()).append("; ");
            }
            
            System.out.println(sb.toString());
        }
        
        long end3 = System.currentTimeMillis();
        System.out.println("third cost time: " + (end3 - end2)/1000.0);
	}

}
