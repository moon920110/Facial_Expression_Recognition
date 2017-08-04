#include <iostream>
#include <fstream>
#include <sstream>
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <linux/input.h>
#include <dirent.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>
#include <wait.h>

#define H_MDIR	"/happy_mp3/"
#define	S_MDIR	"/sad_mp3/"
#define	A_MDIR	"/angry_mp3/"
#define N_MDIR	"/neutral_mp3/"
#define	BUF_LEN	1024
#define	SERVERPORT	9055
#define	SERVERIP	"210.107.236.161"
#define	WIDTH	1024
#define	HEIGHT	600
#define	MAX_MP3	5
#define	MAXBUF	32
#define	HALFBUF	16

using namespace cv;
using namespace std ;

int	volume = 30 ;
int	pid ;
char   	h_list[20][20];
char   	s_list[20][20];
char   	a_list[20][20];
char   	n_list[20][20];
char   	playinx[20];
int   	h_mp3Cnt, s_mp3Cnt, a_mp3Cnt, n_mp3Cnt ;
int		play_num = 0 ;
char   	mp3[16] = {0};
bool	ffff = true ;

typedef struct	sqr_pnt{
	Mat	new_frame;
	Point	lb ;
	Point	tr ;
	int	GUI_num ;
	int	displayFlag ;			// 0 - camera, 1 - default, 2 - display flag
	int	analyzeFlag ;			// 0 - off, 1 - on, 2 - program exit	
	int	expression ;
	int	tmpStatus ;				// currently playing music directory
} pnt ;

void	cleanup_fd(void*) ;
void	cleanup_mmap(void*) ;
void*	FaceDetecting(void*);
void* 	ExpressionRcognition(void*);
void*	ShowDisplay(void*) ;
void*	ScanTouch(void*) ;			// scan touch and show GUI
void	LEDControl(int) ;
void	ChangeClcd(int, int, int, char*) ;
void	DisplayGUI(int) ;
int	Change_to_int(char) ;
int   	MakeList(int);
void	ChangeDot(int) ;
int	Play(char*, int) ;
void	Play_Previous(int) ;
void	Play_Next(int) ;

int		main()
{
	int		thr_id[4] ;
	pthread_t	p_thread[4] ;
	pnt	PP ;
	int	BTN_fd ;
	char	keyValue[9] = {0,0,0,0,0,0,0,0,0} ;
	int	status ;

	//// make music list for each expression
	h_mp3Cnt = MakeList(1);	// happy
   	s_mp3Cnt = MakeList(2);	// sad
   	a_mp3Cnt = MakeList(3);	// angry
   	n_mp3Cnt = MakeList(4);	// neutral

	//// open switch button device
	if((BTN_fd = open("/dev/fpga_push_switch", O_RDONLY))<0){
		printf("button open error\n") ;
		exit(0) ;
	}

	ChangeClcd(0, 0, 0, 0) ;			// start, neutral, no song
	LEDControl(7) ;						// start

	//// init shared variables default
	PP.displayFlag = 2 ;	
	PP.analyzeFlag = 0 ;
	PP.expression = 0 ;
	PP.tmpStatus = 0 ;
	PP.lb.x = 0 ;
	//// create threads
	thr_id[0] = pthread_create(&p_thread[0], NULL, FaceDetecting, (void*)&PP) ;
	thr_id[1] = pthread_create(&p_thread[1], NULL, ExpressionRcognition, (void*)&PP) ;
	thr_id[2] = pthread_create(&p_thread[2], NULL, ShowDisplay, (void*)&PP) ;
	thr_id[3] = pthread_create(&p_thread[3], NULL, ScanTouch, (void*)&PP) ;

	//// infinite loop until program is finished
	while(PP.analyzeFlag >= 0)
	{
		//// check if button is pressed
		read(BTN_fd, &keyValue, sizeof(keyValue)) ;
		usleep(500000) ;

		//// button 0 is analyze on/off
		if(keyValue[0] == 1)
		{
			printf("start/stop analyze\n") ;
			
			PP.analyzeFlag = (PP.analyzeFlag==0 ? 1 :0) ;	// flip the analyze flag
			if(mp3[0] == 0){								// nothing is played
				char c[3] = "No" ;							// print "No" on CLCD
				ChangeClcd(PP.analyzeFlag+2, PP.expression, PP.tmpStatus, c) ;
			}
			else	ChangeClcd(PP.analyzeFlag+2, PP.expression, PP.tmpStatus, mp3) ;
			LEDControl(6 - PP.analyzeFlag) ;				// LED operate
			PP.analyzeFlag == 0 ? ChangeDot(4) : ChangeDot(0) ;	// Dot Mat operate
		}
		//// button 1 is change display
		if(keyValue[1] == 1)
		{
			printf("change display\n") ;
			PP.displayFlag = (PP.displayFlag==0 ? 1 : 0) ;
		}
		if(waitpid(pid, &status, WNOHANG) == pid)			// automatically play next music within same emotion directory
			Play_Next(PP.expression) ;
	}
	
	//// Finish
	ChangeDot(4) ;
	kill(pid, SIGKILL) ;
	printf("music killed\n") ;
	printf("pthread kill\n") ;
	pthread_cancel(p_thread[0]) ;
	pthread_cancel(p_thread[1]) ;
	pthread_cancel(p_thread[2]) ;
	pthread_cancel(p_thread[3]) ;
	printf("start kill\n") ;
	pthread_join(p_thread[0], (void**)&status) ;
	printf("p1 die\n") ;
	pthread_join(p_thread[1], (void**)&status) ;
	printf("p2 die\n") ;
	pthread_join(p_thread[2], (void**)&status) ;
	printf("p3 die\n") ;
	pthread_join(p_thread[3], (void**)&status) ;
	close(BTN_fd) ;
	printf("program quit\n") ;
}

///////////////////////////////////////
//// music play
//// param : music name, emotion status
//// return : currently playing music Pid
int    Play(char *string, int type){
   	char    volumeBuf[4];
   	char   	dir[30];
   	sprintf(volumeBuf, "%d", volume);         // set the current volume
   
	//// set the music directory by current emotion
   	if(type ==1){strcpy(dir, H_MDIR);}
   	else if(type ==2){strcpy(dir, S_MDIR);}
   	else if(type ==3){strcpy(dir, A_MDIR);}
   	else{strcpy(dir, N_MDIR);}

	//// make command
   	char    *argv[] = {"mplayer", "-volume", volumeBuf, "-slave", "-quiet", "-input", "file=/usr/bin/mplayerfifo", "-ao", "alsa:device=hw=1.0","-vo" ,"fbdev:/dev/fb0", strcat(dir, string), (char *)0 };
	//// play new music
   	if ((pid = fork()) == 0) {
      		if (execv("/usr/bin/mplayer", argv) <0) {
         		perror("execv");
         		return 0;
      		}
   	}
   	else {
      		return pid;               // pid of child who plays video
   	}
}

///////////////////////////////////////
//// play previously played music
//// param : emotion status
void   Play_Previous(int type){      // play previous list
   	kill(pid, SIGINT) ;              // stop past video

   	if(type ==1){
      		play_num = (--play_num<0) ? (play_num+h_mp3Cnt) : play_num;
      		strcpy(mp3, h_list[playinx[play_num]]) ;}
   	else if(type ==2){
      		play_num = (--play_num<0) ? (play_num+s_mp3Cnt) : play_num ;
      		strcpy(mp3, s_list[playinx[play_num]]) ;}
  	else if(type ==3){
      		play_num = (--play_num<0) ? (play_num+a_mp3Cnt) : play_num ;
      		strcpy(mp3, a_list[playinx[play_num]]) ;}
   	else{
      		play_num = (--play_num<0) ? (play_num+n_mp3Cnt) : play_num ;
      		strcpy(mp3, n_list[playinx[play_num]]) ;}

   	printf("%s\n", mp3) ;
   	pid = Play(mp3, type) ;

}

///////////////////////////////////////
//// play next music
//// param : emotion status
void   Play_Next(int type){
	if(ffff)	ffff = false ;
   	else 		kill(pid, SIGINT);	// stop past video

   	if(type ==1){
      		play_num = ++play_num % h_mp3Cnt;
      		strcpy(mp3, h_list[playinx[play_num]]);}
   	else if(type ==2){
      		play_num = ++play_num % s_mp3Cnt;
      		strcpy(mp3, s_list[playinx[play_num]]);}
   	else if(type ==3){
      		play_num = ++play_num % a_mp3Cnt;
      		strcpy(mp3, a_list[playinx[play_num]]);}
   	else{
      		play_num = ++play_num % n_mp3Cnt;
      		strcpy(mp3, n_list[playinx[play_num]]);}   
   	printf("%s\n", mp3) ;
   	pid = Play(mp3, type) ;
}

///////////////////////////////////////
//// read touch screen, it works on different thread
//// param : a structure which has shared-variables
void*	ScanTouch(void*	PP)
{
	pnt*	p = (pnt*)PP ;
	int	EVT_fd ;
	int	x, y, status ;
	struct	input_event	ev ;
	int	touchFlag = 0 ;
	int	GUIFlag = 0 ;
	int	showingFlag = 0 ;		// for displaying GUI once
	int	isMain = 0 ;
	bool	isFirst = true ;


	if((EVT_fd = open("/dev/input/event1", O_RDONLY)) < 0){
		printf("event open errer\n") ;
		exit(0) ;
	}

	pthread_cleanup_push(cleanup_fd, (void*)EVT_fd) ;
	while(1)					// touch screen scan
	{
		if(!GUIFlag && p->displayFlag > 0 && isMain==0)	// defualt GUI display
		{
			p->GUI_num = 0 ;
			p->displayFlag = 1 ;
			GUIFlag++ ;
			usleep(100000) ;
		}
		read(EVT_fd, &ev, sizeof(ev)) ;
		if(p->displayFlag > 0)			// touch screen can be activated when GUI is displayed
		{
		
			if(ev.type == 1)		// press or release
				touchFlag++ ;
			else if(ev.type == 3)		// read axis
			{
				if(ev.code == 53)
					x = ev.value ;
				else if(ev.code == 54)
					y = ev.value ;
			}
			if(isMain==0)			// in the main
			{
				if(touchFlag == 1)	// display touched GUI
				{
					if(isFirst)
					{
						p->displayFlag = 1 ;
						isFirst = false ;
					}
					if(x>=0 && x<512 && y>=0 && y<150)		// volume --
						p->GUI_num = 1 ;
					else if(x>=512 && x<WIDTH && y>=0 && y<150)	// volume ++
						p->GUI_num = 2 ;
					else if(x>0 && x<256 && y>=150 && y<450)	// prev
						p->GUI_num = 3 ;
					else if(x>=256 && x<768 && y>=150 && y<450)	// pause
						p->GUI_num = 4 ;
					else if(x>=768 && x<WIDTH && y>150 && y<450)	// next
						p->GUI_num = 5 ;
					else if(x>=0 && x<512 && y>=450 && y<HEIGHT)	// chdir
						p->GUI_num = 6 ;
					else						// exit	
						p->GUI_num = 7 ;
				}

				if(touchFlag == 2)
				{
					isFirst = true ;
					GUIFlag = 0 ;
					if(x>=0 && x<512 && y>=0 && y<150)
					{	
						touchFlag = 0 ;
						system("echo \"volume 0\" > /usr/bin/mplayerfifo") ;
						volume -= 3 ;
						if(volume < 0) volume = 0 ;
						printf("vol:%d\n", volume) ;
					}
					else if(x>=512 && x<WIDTH && y>=0 && y<150)
					{	
						touchFlag = 0 ;
						system("echo \"volume 1\" > /usr/bin/mplayerfifo") ;
						volume+= 3 ;
						if(volume > 100) volume = 100 ;
						printf("vol:%d\n", volume) ;
					}
					else if(x>0 && x<256 && y>=150 && y<450)	// prev
					{	
						touchFlag = 0 ;
						Play_Previous(p->tmpStatus) ;
						usleep(50000) ;
						ChangeClcd((p->analyzeFlag==0?0:1)+2, p->expression, p->tmpStatus, mp3) ;
						printf("prev\n") ;
					}	
					else if(x>=256 && x<768 && y>=150 && y<450){	
						touchFlag = 0 ;
						printf("play/pause\n") ;
						system("echo \"pause\" > /usr/bin/mplayerfifo") ;}
					else if(x>=768 && x<WIDTH && y>150 && y<450)	// next
					{	
						touchFlag = 0 ;
						Play_Next(p->tmpStatus) ;
						usleep(50000) ;
						ChangeClcd((p->analyzeFlag==0?0:1)+2, p->expression, p->tmpStatus, mp3) ;
						printf("next\n") ;
					}	
					else if(x>=0 && x<512 && y>=450 && y<HEIGHT)	// chdir
					{	
						touchFlag = 0 ;
						printf("chdir\n") ;
						p->GUI_num = 13 ;
						p->displayFlag = 1;
						touchFlag = 3 ;
						isMain++ ;
					}	
					else						// exit
					{	
						touchFlag = 0 ;
						ChangeClcd(1, 0, 0, 0) ;
						LEDControl(8) ;
						printf("exit\n") ;
						p->analyzeFlag = -1 ;
					}
				}
			}
			else						// chdir
			{
				if(isMain == 1)
				{
					p->GUI_num = 13 ;		// chdir main
					isMain++ ;
					p->displayFlag = 1;
					usleep(100000) ;
				}
				if(touchFlag == 4)
				{
					if(isFirst)
					{
						p->displayFlag = 1 ;
						isFirst = false ;
					}
					if(x>=0 && x<512 && y>=0 && y<250)		//happy
						p->GUI_num = 8 ;
					else if(x>=512 && x<WIDTH && y>=0 && y<250)	//sad
						p->GUI_num = 9 ;
					else if(x>=0 && x<512 && y>=250 && y<500)	//angry
						p->GUI_num = 10 ;
					else if(x>=512 && x<WIDTH && y>=250 && y<500)	//neutral
						p->GUI_num = 11 ;
					else						//cancel
						p->GUI_num = 12 ;
				}
				else if(touchFlag == 5)
				{
					isFirst = true ;
					touchFlag = 0 ;
					GUIFlag = 0 ;
					if(x>=0 && x<512 && y>=0 && y<250){		//happy
						Play_Next(1) ;
						ChangeClcd((p->analyzeFlag==0?0:1)+2, p->expression, 1, mp3) ;
						p->tmpStatus = 1 ;
						printf("go to happy\n") ;}
					else if(x>=512 && x<WIDTH && y>=0 && y<250){	//sad
						Play_Next(2) ;
						ChangeClcd((p->analyzeFlag==0?0:1)+2, p->expression, 2, mp3) ;
						p->tmpStatus = 2 ;
						printf("go to sad\n") ;}
					else if(x>=0 && x<512 && y>=250 && y<500){	//angry
						Play_Next(3) ;
						ChangeClcd((p->analyzeFlag==0?0:1)+2, p->expression, 3, mp3) ;
						p->tmpStatus = 3 ;
						printf("go to angry\n") ;}
					else if(x>=512 && x<WIDTH && y>=250 && y<500){	//neutral
						Play_Next(0) ;
						ChangeClcd((p->analyzeFlag==0?0:1)+2, p->expression, 0, mp3) ;
						p->tmpStatus = 0 ;
						printf("go to neutral\n") ;}
					// The return does nothing, so we don't need 'else'
					isMain = 0 ;		// return to main
					p->GUI_num = 0 ;
					p->displayFlag = 1 ;
				}
			}
		}
	}

	//// actually it doesn't work... We don't have enough time to implement safe finish
	pthread_cleanup_pop(0) ;
	pthread_exit((void*)0) ;
}

///////////////////////////////////////
//// Display on TFTLCD, it works on different thread
//// param : a structure which has shared-variables
void*	ShowDisplay(void*	PP)
{
	/////////////////
	// gui
	// 0 : default, 1 : volume--, 2 : volume++, 3 : prev, 4 : pause, 5 : next, 6 : chdir, 7 : exit
	// 8 : happy dir, 9 : sad dir, 10 : angry dir, 11 : neutral dir, 12 : return, 13 : chdir main
	/////////////////
	pnt*	p = (pnt*)PP ;
	unsigned short	pix ;
	unsigned short	*fbData ;
	int		fd ;					// TFTLCD fd
	VideoCapture	cap(2);

	if((fd = open("/dev/fb0", O_RDWR)) < 0)			// TFTLCD device open
	{
		printf("fb open error\n") ;
		exit(0) ;
	}
	fbData = (unsigned short*)mmap(0, WIDTH*HEIGHT*2, 
		PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0) ;	// memory mapping for painting
	for(int i=0 ; i<HEIGHT*WIDTH; i++)	*(fbData+i) = 0x0000 ;

	//// I don't know how to use this function..
	pthread_cleanup_push(cleanup_fd, (void*)&fd) ;
	pthread_cleanup_push(cleanup_mmap, (void*)fbData) ;
	while (1)
	{
		cap >> p->new_frame;
		if(p->displayFlag == 0)				// camera display
		{
			if(p->analyzeFlag > 0)
				rectangle(p->new_frame, p->tr, p->lb, Scalar(0, 0, 225), 3, 8, 0);
			for(int k=479 ; k>=0 ; k--)
			{
				for(int j=639 ; j>=0 ; j--)		// Drawing on TFTLCD
				{
					pix = p->new_frame.at<Vec3b>(k, j)[0]/8 + 
						((p->new_frame.at<Vec3b>(k, j)[1]/4)<<5) + 
						((p->new_frame.at<Vec3b>(k,j)[2]/8)<<11) ;
					*(fbData+(k+59)*WIDTH+j+191) = pix ;	// move to center of display
				}
			}
		}
		else
		{
											// GUI display
		}
		{
			if(p->displayFlag == 1)
			{
				Mat	guiImage ;
				switch(p->GUI_num)
				{
					case 0 :
						guiImage = imread("gui_default.jpg") ;
						break ;
					case 1 :
						guiImage = imread("click_volume_down.jpg") ;
						break ;
					case 2 :
						guiImage = imread("click_volume_up.jpg") ;
						break ;
					case 3 :
						guiImage = imread("click_prev.jpg") ;
						break ;
					case 4 :
						guiImage = imread("click_play.jpg") ;
						break ;
					case 5 :
						guiImage = imread("click_next.jpg") ;
						break ;
					case 6 :
						guiImage = imread("click_chdir.jpg") ;
						break ;
					case 7 :
						guiImage = imread("click_exit.jpg") ;
						break ;
					case 8 :
						guiImage = imread("happy.jpg") ;
						break ;
					case 9 :
						guiImage = imread("sad.jpg") ;
						break ;
					case 10 :
						guiImage = imread("angry.jpg") ;
						break ;
					case 11 :
						guiImage = imread("neutral.jpg") ;
						break ;
					case 12 :
						guiImage = imread("cancel.jpg") ;
						break ;
					case 13 :
						guiImage = imread("default.jpg") ;
						break ;
				}
				for(int k=HEIGHT-1 ; k>=0 ; k--)			// Drawing on TFTLCD
				{
					for(int j=WIDTH-1 ; j>=0 ; j--)		
					{
						pix = guiImage.at<Vec3b>(k, j)[0]/8 + 
							((guiImage.at<Vec3b>(k, j)[1]/4)<<5) + 
							((guiImage.at<Vec3b>(k,j)[2]/8)<<11) ;
						*(fbData+k*WIDTH+j) = pix ;
					}
				}
				p->displayFlag += 1 ;
			}
			
		}
	}

	//// actually it doesn't work... We don't have enough time to implement safe finish
	pthread_cleanup_pop(0) ;
	pthread_cleanup_pop(0) ;
	pthread_exit((void*)1) ;
}

///////////////////////////////////////
//// Communicate with Server, it works on different thread
//// param : a structure which has shared-variables
void*	ExpressionRcognition(void* 	PP)		// network communication
{
	pnt*	p = (pnt*)PP ;				// p has point value of face area and video frame
	sleep(2) ;							// for safe operating
	int	s, str_len, imgFd ;				// s is socket
	Mat	croppedImage ;
	struct	sockaddr_in	server_addr ;
	char	buf[BUF_LEN+1] ;
	clock_t start, end ;
	int	past ;
	
	p->expression = 0 ;					// init expression 0(neutral)
	start = clock() ;
	while(1)
	{
		if(p->analyzeFlag > 0)			// analyze mode on
		{
			if(p->analyzeFlag == 1)		// if it is first time, wait 5 seconds for stabilization
			{
				sleep(5) ;
				p->analyzeFlag += 1 ;
			}
			end = clock() ;				// check time
			//// it works every 15 seconds
			if((float)(end-start)/(CLOCKS_PER_SEC) > 15 && p->lb.x != 0)
			{
				printf("send to server %.2f\n", (float)(end-start)/(CLOCKS_PER_SEC)) ;
				memset(buf, 0, sizeof(buf));
				croppedImage = p->new_frame(Rect(p->tr.x, p->tr.y, 	// copy face area to croppedImage
					p->lb.x-p->tr.x, p->lb.y-p->tr.y)) ;	
				resize(croppedImage, croppedImage, Size(224, 224)) ;
				cvtColor(croppedImage, croppedImage, CV_BGR2GRAY) ;
				imwrite("crop.jpg", croppedImage) ;

				if((imgFd = open("crop.jpg", O_RDONLY)) < 0)		// open face image file
				{
					printf("can't open crop.jpg\n") ;
					exit(0) ;
				}
				if((s = socket(AF_INET, SOCK_STREAM, 0)) < 0)		// create socket
				{
					printf("can't create socket\n") ;
					exit(0) ;
				}
				printf("create socket\n") ;
		
				bzero((char*)&server_addr, sizeof(server_addr)) ;
				server_addr.sin_family = AF_INET ;
				server_addr.sin_addr.s_addr = inet_addr(SERVERIP) ;
				server_addr.sin_port = htons(SERVERPORT) ;
				printf("before connect\n") ;
				if(connect(s, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
				{
					printf("can't connect\n") ;
					exit(0) ;
				}
				printf("after connect\n") ;

				while((str_len = read(imgFd, buf, BUF_LEN-1)) != 0)	// send image to server
				{
					write(s, buf, str_len) ;
					memset(buf, 0, sizeof(buf)) ;
				}
				strcpy(buf, "aaaaa") ;						// send EOF message
				write(s, buf, 5) ;
				memset(buf, 0, sizeof(buf)) ;

				str_len = read(s, buf, BUF_LEN-1) ;			// read feedback from the server
				buf[str_len] = '\n';
				printf("Status : %s\n", buf) ;

				p->expression = Change_to_int(buf[0]) ;		// change string to integer
				
				if(past != p->expression)					// emotion is changed
				{
					Play_Next(p->expression) ;				// change the music directory by current emotion and play
					p->tmpStatus = p->expression ;			// set the played directory
					usleep(50000) ;
					ChangeClcd(p->analyzeFlag+1, p->expression, p->tmpStatus, mp3) ; // analyze, emotion, tmpstat, song
					ChangeDot(p->expression) ;
					LEDControl(p->expression) ;				
				}

				close(imgFd) ;
				close(s) ;
				start = clock() ;
				past = p->expression ;						// check current expression
			}
		}
	}
	
	pthread_exit((void*)2) ;
}

///////////////////////////////////////
//// Detect face region using haar cascade, it works on different thread
//// param : a structure which has shared-variables
void*	FaceDetecting(void*	PP)
{
	pnt*	p = (pnt*)PP;							// it's same as TransferImage
	sleep(2) ;
	CascadeClassifier	face_classifier;			// face detecter
	Mat gray ;
	vector<Rect> faces ;
	int	i ;

	if (!face_classifier.load("./haarcascade_frontalface_default.xml"))
	{
		printf("--(!)Error loading\n");
		exit(0) ;
	}
	while(1)
	{
		if(p->analyzeFlag > 0)
		{
			cvtColor(p->new_frame, gray, CV_BGR2GRAY);	// convert captured image to gray scale
			face_classifier.detectMultiScale(gray, faces, 1.5, 5,	// searching face area
				CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,	// have to optimize these factors
				Size(150, 150));

			//// if there is no face then set x pos to 0
			if(faces.size() == 0)
				p->lb.x = 0 ;
			
			//// got the face region
			else
			{
				for (i = 0; i < faces.size(); i++)		// check the points of the detected area
				{
					p->lb.x = faces[i].x + faces[i].width ;
					p->lb.y = faces[i].y + faces[i].height ;
					p->tr.x = faces[i].x ;
					p->tr.y = faces[i].y ;
				}
			}
		}
			
	}
	
	pthread_exit((void*)3) ;
}

///////////////////////////////////////
//// LED controller
//// param : LED control command
void	LEDControl(int	command)
{
	///////////
	// command
	// 1 : Happy, 2 : Sadness, 3 : Angry, 0 : Neutral
	// 5 : analyze on, 6 : analyze off, 7 : start, 8 : Exit
	///////////
	int	LED_fd, i ;
	char 	led ;
	char	tmp ;
	if((LED_fd = open("/dev/fpga_led", O_WRONLY)) < 0){
		printf("Led open error\n") ;
		exit(0) ;
	}

	switch(command)
	{
		case 0 :
			for(i=0 ; i<5 ; i++)
			{
				led = 0xF0 ;
				write(LED_fd, &led, 1) ;
				usleep(200000) ;
				led = 0x0F ;
				write(LED_fd, &led, 1) ;
				usleep(200000) ;
			}
			led = 0x00 ;
			write(LED_fd, &led, 1) ;
			break ;
		case 1 :
			for(i=0 ; i<5 ; i++)
			{
				led = 0xFF ;
				write(LED_fd, &led, 1) ;
				usleep(200000) ; 	// 0.2 seconds
				led = 0x00 ;
				write(LED_fd, &led, 1) ;
				usleep(200000) ;
			}
			break ;
		case 2 :
			led = 0x00 ;
			tmp = 0x80 ;
			for(i=0 ; i<8 ; i++)
			{
				led = led | tmp ;
				tmp = tmp >> 1 ;
				write(LED_fd, &led, 1) ;
				usleep(200000) ;
			}
			led = 0x00 ;
			write(LED_fd, &led, 1) ;
			break ;
		case 3 :
			led = 0x00 ;
			tmp = 0x88 ;
			for(i=0 ; i<4 ; i++)
			{
				led = led | tmp ;
				tmp = tmp >> 1 ;
				write(LED_fd, &led, 1) ;
				usleep(200000) ;
			}
			led = 0x00 ;
			write(LED_fd, &led, 1) ;
			break ;
		case 5 :
			for(i=0 ; i<2 ; i++)
			{
				led = 0xF0 ;
				write(LED_fd, &led, 1) ;
				sleep(1) ;
				led = 0x0F ;
				write(LED_fd, &led, 1) ;
				sleep(1) ;
			}
			led = 0x00 ;
			write(LED_fd, &led, 1) ;
			break ;
		case 6 :
			led = 0xF0 ;
			write(LED_fd, &led, 1) ;
			sleep(2) ;
			led = 0x0F ;
			write(LED_fd, &led, 1) ;
			sleep(2) ;
			led = 0x00 ;
			write(LED_fd, &led, 1) ;
			break ;
		case 7 :
			for(i=0 ; i<2 ; i++)
			{
				led = 0xFF ;
				write(LED_fd, &led, 1) ;
				sleep(1) ;
				led = 0x00 ;
				write(LED_fd, &led, 1) ;
				sleep(1) ;
			}
			break ;
		case 8 :
			led = 0xFF ;
			write(LED_fd, &led, 1) ;
			sleep(2) ;
			led = 0x00 ;
			write(LED_fd, &led, 1) ;
			break ;
	}
	close(LED_fd) ;
}

///////////////////////////////////////
//// CLCD controller
//// param : program status, current expression, playing directory, song title
void	ChangeClcd(int status, int type, int songType, char *song)
{
	//////////////
	// type, songType
	// 0 : neutral(default), 1 : happy, 2 : sadness, 3 : angry
	// status
	// 0 : start, 1 : exit, 2 : don't analyze, 3 : under analyze
	// song
	// default : No song
	//////////////

	char	clcd_buf[33] ;
	int		CLCD_fd ;
	char	emotion[4][15] = {"Neutral ", "Happy ", "Sad ", "Angry "} ;

	//// open CLCD device
	if((CLCD_fd = open("/dev/fpga_text_lcd", O_WRONLY)) < 0){
		printf("clcd open error\n") ;
		exit(0) ;
	}

	memset(clcd_buf, 0, 32) ;
	//// set the phrases
	switch(status)
	{
		case 0 :		// start
			strncat(clcd_buf, "Start", strlen("Start")) ;	
			memset(clcd_buf+strlen(clcd_buf), ' ', MAXBUF-strlen(clcd_buf)) ;	// print on first row
			break ;
		case 1 :		// exit
			strncat(clcd_buf, "Exit Program", strlen("Exit Program")) ;
			memset(clcd_buf+strlen(clcd_buf), ' ', MAXBUF-strlen(clcd_buf)) ;	// print on first row
			break ;
      	case 2 :      // not analyzing
    		strcat(clcd_buf, emotion[songType]);strcat(clcd_buf, song);
         	memset(clcd_buf+strlen(clcd_buf), ' ', HALFBUF-strlen(clcd_buf)) ;	// print on first row
    		strcat(clcd_buf, "None");
         	memset(clcd_buf+strlen(clcd_buf), ' ', MAXBUF-strlen(clcd_buf)) ;	// print on second row
    		break;
      	case 3 :      // under analyzing
         	strcat(clcd_buf, emotion[songType]);strcat(clcd_buf, song);
         	memset(clcd_buf+strlen(clcd_buf), ' ', HALFBUF-strlen(clcd_buf)) ;	// print on first row
    		strcat(clcd_buf, "Do-");strcat(clcd_buf, emotion[type]);	
         	memset(clcd_buf+strlen(clcd_buf), ' ', MAXBUF-strlen(clcd_buf)) ;	// print on second row
         	break ;

	}
	write(CLCD_fd, clcd_buf, MAXBUF) ;	// write on CLCD
	close(CLCD_fd) ;
}

///////////////////////////////////////
//// make play list
//// param : expression type
int   MakeList(int type){                  		// make play list
	int   count = 0 ;
	DIR   *dp ;
	struct   dirent   *dep ;
	int   i ;
	char   D_name[20];

	if(type ==1){strcpy(D_name, H_MDIR);}
	else if(type ==2){strcpy(D_name, S_MDIR);}
	else if(type ==3){strcpy(D_name, A_MDIR);}
	else if(type ==4){strcpy(D_name, N_MDIR);}
	   
	if((dp = opendir(D_name)) == NULL){
		perror("Directory Open Fail") ;
		exit(0) ;
	}
	while(dep = readdir(dp))
	{
		if(strcmp(dep->d_name, ".") == 0){}      // This is itself
		else if(strcmp(dep->d_name, "..") == 0){}   // parent directory
	      	else                  // real files
	      	{
		 //i = dep->d_name[0] - 97 ;      // sorting by the first character
		 	if(type ==1){
		    		strcpy(h_list[count], dep->d_name);
		    		playinx[count] = count;
		 	}
		 	else if(type == 2){
		    		strcpy(s_list[count], dep->d_name);
		    		playinx[count] = count;
		 	}
		 	else if(type == 3){
		    		strcpy(a_list[count], dep->d_name);
		    		playinx[count] = count;
		 	}
		 	else if(type == 4){
		    		strcpy(n_list[count], dep->d_name);
		    		playinx[count] = count;
		 	}
		 	count++;
	      	}
	}
	printf("type %d playlist count %d\n", type, count);
	   
	closedir(dp);
	return    count ;
}

///////////////////////////////////////
//// cahnge string to int
//// param : expression
int   Change_to_int(char string){
   	int num;
   
   	if(string == 'H')
      		num = 1;
   	
   	else if(string == 'S')
      		num = 2;
   	
   	else if(string == 'A')
      		num = 3;
   	
   	else
      		num = 0;

   	return num; // 1 - happy 2 - sad 3 - angry 0 - neutral

}

///////////////////////////////////////
//// Dot matrix controller
//// param : expression
void   ChangeDot(int   type){
   	char   	h_dot[10] = {0x00, 0x10, 0x22, 0x11, 0x01, 0x01, 0x11, 0x22, 0x10, 0x00};
   	char   	s_dot[10] = {0x00, 0x20, 0x11, 0x22, 0x02, 0x02, 0x22, 0x11, 0x20, 0x00};
   	char   	a_dot[10] = {0x00, 0x20, 0x11, 0x11, 0x01, 0x01, 0x11, 0x11, 0x20, 0x00};
   	char   	n_dot[10] = {0x00, 0x20, 0x22, 0x21, 0x01, 0x01, 0x21, 0x22, 0x20, 0x00};
	char	dft[10] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
   	char   	em_dot[10];
	int	Dot_fd ;

	//// open dot matrix device
	if((Dot_fd = open("/dev/fpga_dot", O_RDWR)) < 0){
		printf("Dot open error\n") ;
		exit(0) ;
	}

   	memset(em_dot, 0, 10);
   	switch(type){
      		case 0: memcpy(em_dot, n_dot, 10);break;
      		case 1: memcpy(em_dot, h_dot, 10);break;
      		case 2: memcpy(em_dot, s_dot, 10);break;
      		case 3: memcpy(em_dot, a_dot, 10);break;
		case 4: memcpy(em_dot, dft, 10);break;
   	}
   	write(Dot_fd, &em_dot, 10) ;
	close(Dot_fd) ;
}

///////////////////////////////////////
//// clear mapped memory
//// param : mapped memory pointer
void	cleanup_mmap(void*	data)
{
	unsigned short	*d = (unsigned short*)data ;
	munmap(d, WIDTH*HEIGHT*2) ;
	printf("***memory unmap***\n") ;
}

///////////////////////////////////////
//// close fd(device)
//// param : device fd
void	cleanup_fd(void*	data)
{
	int*	d = (int*)data ;
	close(*d) ;
	printf("***free fd***\n") ;
}
