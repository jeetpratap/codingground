import java.io.File;
import java.io.IOException;

public class NewFileCreation {
   public static void main(String[] args) {
      try {
         File file = new File("E:\\d\\Mainframe\\JAVA\\Java Programs\\Files\\myfile.ppt");
         
         if(file.createNewFile())System.out.println("Success!");
         else System.out.println ("Error, file already exists.");
      }
      catch(IOException ioe) {
         ioe.printStackTrace();
      }
   }
}