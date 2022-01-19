  <?php 
  	session_start();
   ?>

  <!doctype html>
   <html lang="en">
   <head>
   	<!-- Required meta tags -->
   	<meta charset="utf-8">
   	<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

   	<!-- Bootstrap CSS -->
   	<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-BmbxuPwQa2lc/FVzBcNJ7UAyJxM6wuqIj61tLrc4wSX0szH/Ev+nYRRuWlolflfl" crossorigin="anonymous">
   	<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins&display=swap">
   	<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.8.1/css/all.css" integrity="sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf" crossorigin="anonymous">
   	<title>News Summarization</title>	

   	<style type="text/css">
   	#judul{
   	   	font-family: 'Poppins', sans-serif;
       	padding: 8px;
        	text-align: center;
        	width: 100%;
     }
     #title{
     	font-size: 24px;
     	color: white;
     }
     #nav-color{
      	background-color: #E7E7E7
     }
   	.tp{
   		font-weight: normal;
   	}
   	.spin{
   		width: 5rem;
   		height: 5rem;
   	}
   	#loading{
   		width: 100%;
   		height: 100%;
   		margin:auto;
   		top: 0;
   		left: 0;
   		display: block;
   	}
   	.hidden-text{
   		visibility: hidden;
   	}
   	.tooltip {
   		position: relative;
   		display: inline-block;
   		border-bottom: 1px dotted black;
   	}

   	.tooltip .tooltiptext {
   		visibility: hidden;
   		width: 120px;
   		background-color: white;
   		color: #fff;
   		text-align: center;
   		border-radius: 6px;
   		padding: 5px 0;
   		position: absolute;
   		z-index: 1;
   	}
   	.tooltip:hover .tooltiptext {
   		visibility: visible;
   	}

   </style>
</head>

<body>

	<?php
	if (isset($_SESSION['message']) && $_SESSION['message'])
	{
		echo '<script>alert("'.$_SESSION['message'].'");</script>';
		unset($_SESSION['message']);
	}
	?>
	<nav class="navbar navbar-expand-lg navbar-center navbar-dark bg-danger" id="nav-color">
		<div class="container-fluid">
			<div class="row" id="judul">
				<span class="navbar-text mx-auto" href="#"><b id="title">Indonesian News Text Summarizer</b></span>
			</div>
		</div>
	</nav>
	<br><br>

	<div class="row justify-content-center align-self-center mt-5" id="loading">
		<div class="col mx-auto my-auto">
			<div class="text-center">
				<div class="spinner-border spin" role="status">
					<span class="sr-only">Loading...</span>
				</div>
			</div>
		</div>
	</div>


	<?php if (isset($_SESSION['flag'])) { ?>
		<div class="container">
			<div class="row">
				<div class="col-6">
					<h5><b>Original News</b></h5>
					<?php if (isset($_SESSION['original_news'])) { echo $_SESSION['original_news']; unset($_SESSION['original_news']); }?>
				</div>
				<div class="col-6">
					<h5><b>Summary Result</b></h5>
					<?php if(isset($_SESSION['result'])){  
						foreach ($_SESSION['result'] as $key => $value) {
							echo $value."<br><br>";
						}
						unset($_SESSION['result']); 
					}?>
				</div>
			</div>
		</div>
		<?php } else { ?>
			<div class="container" id="content">
				<form method="POST" name="summarizer_form" action="result.php" enctype="multipart/form-data">
					<div class="row">
						<div class="col-11 mb-3">
							<label for="news_url" class="form-label"><b>News article URL</b></label>
							<input type="url" class="form-control" id="news_url" name="news_url" placeholder="https://www.cnbcindonesia.com/news/20210823170354-4-270564/kejar-target-rp-1200-t-ini-segudang-kebijakan-pajak-2022">
						</div>
					</div>
					<br>
					<div class="row">
						<div class="col-11 mb-3">
							<label for="uploadedNewsFile" class="form-label"><b>News file</b></label>
							<input accept=".txt" class="form-control" type="file" id="uploadedNewsFile" name="uploadedNewsFile">
						</div>
					</div>
					<br>
					<div class="row">
						<div class="col-11 mb-3">
							<label for="inputPercentages" class="form-label"><b>Percentages of summary</b></label>
							<input type="number" class="form-control" id="inputPercentages" name="inputPercentages" min="0" max="100" placeholder="50">
						</div>
						<div class="col-1 mb-3 mt-5">
							<p>%</p>
						</div>
					</div>
					<div class="row">
						<div class="col text-center">
							<input type="submit" class="btn btn-lg btn-secondary" name="summarizeBtn" value="Summarize" onclick="return validateForm()">
						</div>
					</div>
				</form>
			</div>
		<?php } unset($_SESSION['flag']); ?>

	

	<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0=" crossorigin="anonymous"></script>
	<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
	<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js" integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI" crossorigin="anonymous"></script>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-confirm/3.3.4/jquery-confirm.min.js" integrity="sha256-Ka8obxsHNCz6H9hRpl8X4QV3XmhxWyqBpk/EpHYyj9k=" crossorigin="anonymous"></script>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-notify/0.2.0/js/bootstrap-notify.js" integrity="sha256-lY8OdlU6kUK/9tontLTYKJWThbOuSOlWtbINkP0DLKU=" crossorigin="anonymous"></script>

	<script>
		$(window).on('load',function() {
			$('#loading').hide();
			$('#content').show();
		});

		$(document).ready(function(){
			$('form').submit(function() 
			{
				$('#loading').show();
				$('#content').hide();
			});
		});
		function validateForm() {
			var url = document.forms["summarizer_form"]["news_url"].value;
			var file = document.forms["summarizer_form"]["uploadedNewsFile"].value;
			var percentages = document.forms["summarizer_form"]["inputPercentages"].value;
			if ((url == "" && file == "") || percentages==""){
				alert("Please fill news URL or file with percentages field");
				return false;
			}
			return true;
		}
	</script>

</body>

</html>