The entire code is based on the research mentioned in the paper Sanchit2014 Estimating Floor Regions in Cluttered Indoor Scenes from First Person Camera View [https://cvit.iiit.ac.in/images/ConferencePapers/2014/Sanchit2014Estimating.pdf]



ViShruti Simulation Version Notes

Version Number: V4
1.	Created On  : 11th Novemeber 2013
2.	Modified On : 1st March 2014
3. 	Main Features:
	a) Staircasing for ISI in ascending sequence starting from 200,25,50,100,200,300,400,500
	b) Three phase design.
	C) Phase 1 - Working memeory with Automatic selection of ISI followed by 5 schemas for 4 dir and 8 dir WM with 2-8 length varied cues
	d) Phase 2 - Training and Testing Module with 80% accuracy threshold in three consecutive maps, 4 dir train, 8 dir train followed by 4 dir testing and 8 dir testing.
	e) Phase 3 - Working Memory with previously selected ISI followed by 5 schemas for 4 dir and 8 dir WM with 2-8 length varied cues
	f) Keyboard control for response
	g) Recording data both in csv format and google excel sheet.
	h) Audio Instruction in telugu added.
	i) English instruction shown in dilaogue prompts.
	j) AB design for WM phases


4.	Experimented on 26 Sighted participant for four group of Audio error feedback, Visual Error Feedback, Unsupervised Learning and No Training
5.	Submited to Spatical Cognition 2014 on 8 March 2014.



Version Number: V5
1.	Created On  : 12th May 2014
2.	Modified On : 15th May 2014
3. 	Main Features:
	a) ABAB design for WM phases. To check fatigue and break boredom, increased no of maps for both 4 direction and 8 direction to 6 each. 3 for A1, 3 for A2, 3 for B1 and 3 for B2 where A and B represents 4 direction and B 8 Direction blocks respectively.
	b) randomized stair casing for selection of ISI interval with fixed stimuli (#3) per trial. 10 trials for each (25,50,100,200,300,400 and 500) ms intervals. Minimum ISI for which recall is >80% is selected as Best ISI on which the rest of the experiment is conducted. In case, there are two ISI with recall >80% and are 100 ms apart then again 5-5 trial are presented to select the best among the two.
		case 1) For eg. 100,500 comes to the candidate ISI, then in order to select the best ISI, 5-5 trial for both are given and then the minimum with recall >80 is selected among the two.
                case 2) For eg. 100,200,400 , Best ISI = 100.
	c) Demo added before each block. Each Demo will have Consistent 2 Minutes gaps between each phase and between each switch
	d) changes in database recordings,  implemented node js to store per trial input/response
	e) Joystick controls added
