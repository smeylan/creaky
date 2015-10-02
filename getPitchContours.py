# -*- coding: utf-8 -*-
"""
Get the pitch contour for each utterance; aggregate by speaker or by phrase
"""
import os.path, pandas, glob, numpy as np, re, pdb, audiolabel 
import matplotlib.pyplot as plt
from numpy import nanmean
from matplotlib.backends.backend_pdf import PdfPages

def getPitchContourForSubject(random_id, analysisDir, vre, nsteps, utterances):
	'''enumerate the uttterances made by a subject and process each separately; return a dataframe with each sample for each *phone* as a separate record'''
	print('Processing subject '+str(random_id))
	globtext = os.path.join(analysisDir, 'voicesof_data','*{}*.wav').format(random_id)  # find the wav files for the speaker
	files = glob.glob(globtext)
	df =  pandas.concat([getPitchContour(file, vre, nsteps, analysisDir) for file in files])	
	df['random_id'] = random_id
	df['utterance'] = [utterances[x] for x in df['phraseIndex']]
	return(df)


def getPitchContour(soundfile, vre, nsteps, analysisDir):	
	(basename,ext) = os.path.splitext(soundfile)
	basename = os.path.basename(basename)
	directory = os.path.dirname(soundfile)
	phrase = basename.split('_')[1]  # get the phrase number from the file name
	phrase = int(phrase)-1

	tg = soundfile.split('.')[0] + '.TextGrid'   # the name of the associated TextGrid
	if (not os.path.isfile(tg)):  # check to see if the textgrid exists
		error = 'no textgrid file: {}'.format(tg)  
		return(pandas.DataFrame({'t1':None,'f0' : None,'f1' : None,'f2' : None,'f3' : None,'f4' : None,'phoneSampleIndex': None,'phoneIndex':None,'p' : None,'stress': None,'word': None,'vowel': False,'error': error, 'phraseIndex':phrase,'file':soundfile}, index=[0]))
	pm  = audiolabel.LabelManager(from_file=tg, from_type='praat')  # read the textgrid

	error = None
	if phrase > 5:
		error = ('Omitting the math phrase')		
		return(pandas.DataFrame({'t1':None,'f0' : None,'f1' : None,'f2' : None,'f3' : None,'f4' : None,'phoneSampleIndex': None,'phoneIndex':None,'p' : None,'stress': None,'word': None,'vowel': False,'error': error, 'phraseIndex':phrase,'file':soundfile}, index=[0]))
	try:   #test for 'phone' tier in the textgrid file
		pm.tier('phone')
	except IndexError:
		error = 'no phone tier in: {}'.format(tg)
		return(pandas.DataFrame({'t1':None,'f0' : None,'f1' : None,'f2' : None,'f3' : None,'f4' : None,'phoneSampleIndex': None,'phoneIndex':None,'p' : None,'stress': None,'word': None,'vowel': False,'error': error, 'phraseIndex':phrase,'file':soundfile}, index=[0]))
	
	formants_name = soundfile.split('.')[0] + '.fb'  # the name of the associated formants file
	if (not os.path.isfile(formants_name)):  # check to see if it exists
		error = 'no formants file: {}'.format(formants_name)
		return(pandas.DataFrame({'t1':None,'f0' : None,'f1' : None,'f2' : None,'f3' : None,'f4' : None,'phoneSampleIndex': None,'phoneIndex':None,'p' : None,'stress': None,'word': None,'vowel': False,'error': error, 'phraseIndex':phrase,'file':soundfile}, index=[0]))
	# read the formants file
	
	formants = audiolabel.LabelManager(from_file=formants_name, from_type='table',  fields_in_head=False, fields = "t1,rms,f1,f2,f3,f4,f0")
	formants.scale_by(0.001)  # convert time column to seconds    
     # find all the vowel labels in the phone tier and process them  

	sampleStore = []
	vowels = []
	matches = []
	for v,m in pm.tier('phone').search(vre, return_match=True): #will use this to determine whether the phone is a vowel or not
		vowels.append(v)
		matches.append(m)
	phoneIndex = -1 #index of the phone
	phoneSampleIndex = -1 #index of the sample of the phone. only 1 sample for a consonant		
	for p, m in pm.tier('phone').search('^((?!sil).)*$', return_match=True): #iterate through non-syllables		
		if p.text == 'sp':
			continue
		phoneIndex += 1
		word=pm.tier('word').label_at(p.center).text
		if p in vowels: 						
			dur = p.duration  # "v" is a textgrid label object, one of it's properties is duration
			if dur < 0.029 or (dur > 1 and p.t1 != 0): # don't look at really short vowels
				continue                
			vStress = p.text	
			vowel =  vStress[0:-1] # these lines take a label like IY1 and split it
			stress = vStress[-1]  # into "IY" and "1" 
			if stress != "1":  # only look at stressed vowels  (1=primary, 2=secondary, 0=unstressed)
				continue			
			if vowel =='AE' and (word=="HAND" or word=="STAND"):
				vowel = "AEN"
			inc = dur/(nsteps)  # extract formant measurements from 25%, 50% and 75% of vowel duration   
			
			for i in range(1,nsteps):  # find the three locations in the vowel
				phoneSampleIndex += 1
				ttime = p.t1 + (inc*i)    # time of step
				meas = formants.labels_at(ttime)
				rdict = {'t1':meas.f0.t1,
							'f0' : float(meas.f0.text),
							'f1' : float(meas.f1.text),
							'f2' : float(meas.f2.text),
							'f3' : float(meas.f3.text),
							'f4' : float(meas.f4.text),
							'phoneSampleIndex': phoneSampleIndex,
							'phoneIndex':phoneIndex,
							'p' : p.text,
							'stress': stress,
							'word': word,
							'vowel': True,
							'error': error}
				sampleStore.append(rdict)
				#!!! AEN is treated iffky
		else: #if p is a consonant			
			phoneSampleIndex += 1 #increment once
			rdict = {'t1':None,
							'f0' : None,
							'f1' : None,
							'f2' : None,
							'f3' : None,
							'f4' : None,
							'phoneSampleIndex': phoneSampleIndex,
							'phoneIndex':phoneIndex,
							'p' : p.text,
							'stress': None,
							'word': word,
							'vowel': False,
							'error': error}
			sampleStore.append(rdict)		
	pdf = pandas.DataFrame(sampleStore)
	pdf['phraseIndex'] = phrase
	pdf['file'] = soundfile
	pdf['error'] = error
	return(pdf)
	

def getUtteranceContourPlot(results, pI, utterances, pp):
	singlePhrase = results.loc[(results['phraseIndex'] == pI)] 

	rpi = singlePhrase.groupby(['phoneIndex','p','gender']).agg({'f0': nanmean}).reset_index() 	
	rpi_label = singlePhrase.groupby(['phoneIndex','p']).agg({'f0': nanmean}).reset_index()
	rpi_label_collapsed = rpi_label.groupby(['phoneIndex']).agg({'p': lambda x: "%s" % '|'.join(x)}).reset_index()


	ymax = np.nanmax(rpi['f0'])[0] *1.1
	ymin = np.nanmax(rpi['f0'])[0] *.9
	#!!! 0s are bringing this down; !!!F0s are too high
	xmax = np.nanmax(rpi['phoneIndex'])[0] 


	fig = plt.figure(figsize=(14, 6))
	ax = fig.add_subplot(1,1,1)                                                      
	#need the labels

	major_ticks = range(0,xmax+1)
	ax.set_xticks(major_ticks)      # we have five points per vowel         
	ax.set_xticklabels(rpi_label_collapsed['p'])
	ax.tick_params(axis='x', which='major', labelsize=7)

	for gender, color, label in [('m', 'r--','men'),('f', 'b-', 'women')]:
		genderDF = rpi.loc[rpi['gender'] == gender]
		genderDF = genderDF.dropna() 
		plt.plot(genderDF['phoneIndex'],genderDF['f0'],color, label=label)
	#	plt.plot(genderDF['phoneIndex'],genderDF['f0'],color, label=label)
		#ordering needs to be a function of the time			
	plt.title(utterances[pI])
	plt.ylim(0,ymax)
	plt.xlim(0,xmax)
	plt.legend(loc='lower left')
	plt.xlabel('Phone')
	plt.ylabel('F0')
	plt.grid(True)
	plt.savefig(pp, format='pdf')

def subjectF0Hist(resultsNoZerosMerged, random_id):
	singleSubject = resultsNoZerosMerged.loc[resultsNoZerosMerged['random_id'] ==random_id].dropna(subset=['f0'])
	
	plt.hist(list(singleSubject['f0']))
	plt.title("F0 for participant "+str(random_id))
	plt.xlabel("F0")
	plt.ylabel("Frequency")
	plt.show()


if __name__ == '__main__':
	analysisDir = "/media/sf_Box_Sync/ling290-f2015/"
	vre = re.compile("^(?P<vowel>AA|AE|AH|AO|AW|AXR|AX|AY|EH|ER|EY|IH|IX|IY|OW|OY|UH|UW|UX)(?P<stress>\d)?$")
	nsteps = 4

	multicore = True

	subjects = pandas.io.parsers.read_table(os.path.join(analysisDir, 'subjects.txt'))

	utterances = ["Go Bears", "Dawn found it odd that Judd did a hand stand.",
	"She had your dark suit in greasy wash water all year.",
	"Who said you should hold such an awkward pose?",
	"Don was awed by the hat rack.",
	"This wheel's red spokes show why mud is no boon.",
	"Ten plus one equals eleven and two plus six equals eight."]

	if multicore:
		#increase the available core count under "processors" in the VM settings if you want to parallelize this. Should have an approximately linear speedup with the number of logical cores that the VM can use.
		#to install joblib: sudo easy_install-2.7 joblib
		import joblib, multiprocessing
		from joblib import delayed, Parallel
		subjectDFs = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(getPitchContourForSubject)(x, analysisDir, vre, nsteps, utterances) for x in subjects['random_id'])
	else:
		subjectDFs = [getPitchContourForSubject(x, analysisDir, vre, nsteps, utterances) for x in subjects['random_id']]		
	results = pandas.concat(subjectDFs)

	#remove samples with f0 of 0
	resultsNoZeros = results.loc[results['f0'] != 0]
	
	resultsNoZerosMerged = pandas.merge(resultsNoZeros, subjects)

	pp = PdfPages('pitchContours.pdf')
	for i in range(0,6):
		getUtteranceContourPlot(resultsNoZerosMerged, i, utterances, pp)
	pp.close()
	
	subjectF0Hist(resultsNoZerosMerged, 2328645)
