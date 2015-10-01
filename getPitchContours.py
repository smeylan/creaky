# -*- coding: utf-8 -*-
"""
Get the pitch contour for each utterance; aggregate by speaker or by phrase
"""
import os.path, pandas, glob, numpy as np, re, pdb, audiolabel 

def getPitchContourForSubject(random_id, analysisDir, vre, nsteps, utterances):
	'''enumerate the uttterances made by a subject and process each separately; return a dataframe with each utterance as a separate record'''
	print('Processing subject '+str(random_id))
	globtext = os.path.join(analysisDir, 'voicesof_data','*{}*.wav').format(random_id)  # find the wav files for the speaker
	files = glob.glob(globtext)
	
	df =  pandas.DataFrame([getPitchContour(file, vre, nsteps, analysisDir) for file in files])	
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
		return({'formants':None, 'phraseIndex':phrase, 'file':soundfile,'error':error})
	pm  = audiolabel.LabelManager(from_file=tg, from_type='praat')  # read the textgrid

	error = None
	if phrase > 5:
		error = ('Omitting the math phrase')		
		return({'formants':None, 'phraseIndex':phrase, 'file':soundfile,'error':error})
	try:   #test for 'phone' tier in the textgrid file
		pm.tier('phone')
	except IndexError:
		error = 'no phone tier in: {}'.format(tg)
		return({'formants':None, 'phraseIndex':phrase, 'file':soundfile,'error':error})
	
	formants_name = soundfile.split('.')[0] + '.fb'  # the name of the associated formants file
	if (not os.path.isfile(formants_name)):  # check to see if it exists
		error = 'no formants file: {}'.format(formants_name)
		return({'formants':None, 'phraseIndex':phrase, 'file':soundfile,'error':error})
	# read the formants file
	
	formants = audiolabel.LabelManager(from_file=formants_name, from_type='table',  fields_in_head=False, fields = "t1,rms,f1,f2,f3,f4,f0")
	formants.scale_by(0.001)  # convert time column to seconds    
     # find all the vowel labels in the phone tier and process them  

	vowelF0s = [] 
	vowelTimes = []         
	for v, m in pm.tier('phone').search(vre, return_match=True):
		dur = v.duration  # "v" is a textgrid label object, one of it's properties is duration
		if dur < 0.029 or (dur > 1 and v.t1 != 0): # don't look at really short vowels
			continue                
		vowel = m.group('vowel')  # these lines take a label like IY1 and split it
		stress = m.group('stress')  # into "IY" and "1" 
		if stress != "1":  # only look at stressed vowels  (1=primary, 2=secondary, 0=unstressed)
			continue
		if vowel =='AE':                    
			word=pm.tier('word').label_at(v.center).text
			if word=="HAND" or word=="STAND":
				vowel = "AEN"
		inc = dur/(nsteps)  # extract formant measurements from 25%, 50% and 75% of vowel duration   
		        		
		for i in range(1,nsteps):  # find the three locations in the vowel
			ttime = v.t1 + (inc*i)    # time of step
			meas = formants.labels_at(ttime)
			vowelF0s.append(float(meas.f0.text))
			vowelTimes.append(float(meas.f0.t1))
	
	return({'formants': vowelF0s, 'times': vowelTimes, 'phraseIndex':phrase, 'file':soundfile,'error':error})
			#!!!what do we need to pull out of meas? how many samples


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




