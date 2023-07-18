
<?php
    #echo "<body style='background-color:cyan'>";

	if(isset($_POST['submit']))
	{   $date= $_POST['datepicker'];
       
		$arr = $date;
		$resStr = str_replace('/', '-', $arr);
		
		$OriginalString = $resStr;
	
		$arr = explode("-",$OriginalString);

 		$res = ($arr[1])."-".($arr[0])."-".($arr[2]);
		print_r("<u><h3><center>Predictions for $res</center></u></h3");
		$r = shell_exec("python predict.py");
		echo "$r\n";
        $imgUrl = "predict-image.png"; 
		
		
	}
	

?>    
<center><img width= 900 height=700 src="<?= $imgUrl;?>"/></center>


 

