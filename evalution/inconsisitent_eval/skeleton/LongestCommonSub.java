package skeleton;

import java.io.IOException;
import java.util.List;

import preprocess.HybridTokenizer;

public class LongestCommonSub {
	public static void main(String[] args) throws IOException {  
        // TODO Auto-generated method stub  
        String str1 = "#GazaUnderAttack #Hamas rocket lands in #Gaza, kills baby, wound 4. #Stoptherockets";  
        String str2 = "#GazaUnderAttack #Hamas rocket hits #gaza, kills baby, wounds 4."; 

        
        HybridTokenizer hybridTokenizer = new HybridTokenizer();
	  	List<String> filteredTokenList1 = hybridTokenizer.tokenize(str1);
	  	List<String> filteredTokenList2 = hybridTokenizer.tokenize(str2);
	  	String[] aa = new String[filteredTokenList1.size()];
	  	filteredTokenList1.toArray(aa);
	  	String[] bb = new String[filteredTokenList2.size()];
	  	filteredTokenList2.toArray(bb);
	  	
	  	
        
//        String[] aa=Eng_Similarity.getTokens(str1).split(" ");
//        String[] bb=Eng_Similarity.getTokens(str2).split(" ");
        System.out.println(aa.length+" "+bb.length);
//        String[] aa = str1.split(" "); 
//    	String[] bb = str2.split(" "); ;
        //计算lcs递归矩阵  
        int[][] re = longestCommonSubsequence(aa, bb);  
        //打印矩阵  
//        for (int i = 0; i <= aa.length; i++) {  
//                for (int j = 0; j <= bb.length; j++) {  
//                        System.out.print(re[i][j] + "   ");  
//                }  
//                System.out.println();  
//        }  
//  
//        System.out.println();  
//        System.out.println();  
        //输出LCS  
        print0(re, aa, bb, aa.length, bb.length);  
    }  
  
    // 假如返回两个字符串的最长公共子序列的长度  
    public static int[][] longestCommonSubsequence(String[] str1, String[] str2) {  
        
    	int[][] matrix = new int[str1.length + 1][str2.length + 1];//建立二维矩阵  
        // 初始化边界条件  
        for (int i = 0; i <= str1.length; i++) {  
                matrix[i][0] = 0;//每行第一列置零  
        }  
        for (int j = 0; j <= str2.length; j++) {  
                matrix[0][j] = 0;//每列第一行置零  
        }  
        // 填充矩阵  
        for (int i = 1; i <= str1.length; i++) {  
                for (int j = 1; j <= str2.length; j++) {  
//                        if (str1.charAt(i - 1) == str2.charAt(j - 1)) { 
                	if (str1[i-1].equals(str2[j - 1])) { 
                                matrix[i][j] = matrix[i - 1][j - 1] + 1;  
                        } else {  
                                matrix[i][j] = (matrix[i - 1][j] >= matrix[i][j - 1] ? matrix[i - 1][j]  
                                                : matrix[i][j - 1]);  
                        }  
                }  
        }  
        return matrix;  
    }  
    //根据矩阵输出LCS  
    public static List<String> print(int[][] opt, String[] X, String[] Y, int i, int j,List<String> str) {  
        if (i == 0 || j == 0) {  
                return str;  
        }  
        if (X[i - 1].equals(Y[j - 1])) {  
                print(opt, X, Y, i - 1, j - 1,str);  
                str.add(X[i - 1]);
                //System.out.print(X[i - 1]+" ");  
        } else if (opt[i - 1][j] >= opt[i][j]) {  
                print(opt, X, Y, i - 1, j,str);  
        } else {  
                print(opt, X, Y, i, j - 1,str);  
        } 
        return str;
    }  
    
    public static void print0(int[][] opt, String[] X, String[] Y, int i, int j) {  
        if (i == 0 || j == 0) {  
                return ;  
        }  
        if (X[i - 1].equals(Y[j - 1])) {  
                print0(opt, X, Y, i - 1, j - 1);  
                System.out.print(X[i - 1]+" ");  
        } else if (opt[i - 1][j] >= opt[i][j]) {  
                print0(opt, X, Y, i - 1, j);  
        } else {  
                print0(opt, X, Y, i, j - 1);  
        } 
    }  
}
