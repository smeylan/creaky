# -*- coding: utf-8 -*-
"""
tabulate creak by speaker

"""
import os.path, pandas, glob, numpy as np, re, pdb 
import audiolabel  # Ronald Sprouse library for reading Praat textgrids (among other things)

def getScoresForSubject(random_id, vre, analysisDir, nsteps, utterances):
	print('Processing subject '+str(random_id))
	globtext = os.path.join(analysisDir, 'voicesof_data','*{}*.wav').format(random_id)  # find the wav files for the speaker #!!! not sure what .format does in this context
	files = glob.glob(globtext)
	
	df =  pandas.DataFrame([getCreakScore(file, vre, analysisDir, nsteps) for file in files])
	df['creakSum'] = [np.sum(x) for x in df['creakScore']] 
	df['sampleSum'] = [np.sum(x) for x in df['samples']] 
	df['random_id'] = random_id
	df['utterance'] = [utterances[x] for x in df['phraseIndex']]
	return(df)

def getCreakScore(soundfile,  vre, analysisDir, nsteps):
	'''get the creak vector for a sentence, with metadata''' 
	#!!! do we want to get back other metadata?

	#nsteps: divide the vowel duration into 6 chunks - five measurements per vowel
	#analysisDir: path to look for the data at
	data = np.zeros(70,dtype=np.int) #time points for the phrase. Why 70?
	counts = np.zeros(70,dtype=np.int) #time points for the phrase. Why 70?
	
	ntime = 0
	(basename,ext) = os.path.splitext(soundfile)
	basename = os.path.basename(basename)
	directory = os.path.dirname(soundfile)
	phrase = basename.split('_')[1]  # get the phrase number from the file name
	phrase = int(phrase)-1
	
	error = None
	if phrase > 5:
		error = ('Omitting the math phrase')		
		return({'creakScore':None, 'samples':None, 'phraseIndex':phrase, 'file':soundfile,'error':error})

	tg = os.path.join(directory, basename+'.TextGrid')   # the name of the associated TextGrid
	if (not os.path.isfile(tg)):  # check to see if the textgrid exists
		 error = 'no textgrid file: {}'.format(tg)  
		 return({'creakScore':None, 'samples':None, 'phraseIndex':phrase, 'file':soundfile,'error':error})
	else:	 
		pm  = audiolabel.LabelManager(from_file=tg, from_type='praat')  # read the textgrid

	try: #test for 'phone' tier in the textgrid file
		pm.tier('phone')
	except IndexError:
		error = 'no phone tier in: {}'.format(tg)
		return({'creakScore':None, 'samples':None, 'phraseIndex':phrase, 'file':soundfile,'error':error})		

	creak_name = os.path.join(directory, basename+'.creak')  # the name of the associated formants file
	if (not os.path.isfile(creak_name)):  # check to see if it exists
		error = 'no formants file: {}'.format(creak_name)
		return({'creakScore':None, 'samples':None, 'phraseIndex':phrase, 'file':soundfile,'error':error})

	# read the creak file
	creakfile = audiolabel.LabelManager(from_file=creak_name, 
		from_type='table', fields_in_head=False, fields = "t1,score,creak")
                                  
	for v, m in pm.tier('phone').search(vre, return_match=True):
		dur = v.duration
	    
		vowel = m.group('vowel')  # these lines take a label like IY1 and split it
		stress = m.group('stress')  # into "IY" and "1" 

		inc = dur/(nsteps)  # extract formant measurements from 25%, 50% and 75% of vowel duration           

		for i in range(1,nsteps):  # find the five locations in the vowel
			# this is a for loop because the creak file has relative, not absolute times
			ttime = v.t1 + (inc*i)    # time of step

			meas = creakfile.labels_at(ttime)  # extract the formant measurements at this time

			data[ntime] += int(meas.creak.text)
			counts[ntime] += 1
			ntime += 1
	return({'creakScore':data, 'samples':counts, 'phraseIndex':phrase, 'file':soundfile,'error':None})


if __name__ == '__main__':
	analysisDir = "/media/sf_Box_Sync/ling290-f2015/"
	multicore = True
	vre = re.compile("^(?P<vowel>AA|AE|AH|AO|AW|AXR|AX|AY|EH|ER|EY|IH|IX|IY|OW|OY|UH|UW|UX)(?P<stress>\d)?$") # a regular expression used to recognize and decode the vowel labels
	nsteps = 6 # divide the vowel duration into 6 chunks - five measurements per vowel

	#load the subjects table
	subjects = pandas.io.parsers.read_table(os.path.join(analysisDir, 'subjects.txt'))  # read in subject information
	CAsubjects = pandas.concat([subjects[subjects.adminAreaLevel1=="California"] , subjects[subjects.adminAreaLevel1=="CA"]]) #filter just to Californians
	#random_ids link subjects to their files

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
		subjectDFs = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(getScoresForSubject)(x, vre, analysisDir, nsteps, utterances) for x in CAsubjects['random_id'])
	else:
		subjectDFs = [getScoresForSubject(x, vre, analysisDir, nsteps, utterances) for x in CAsubjects['random_id']]		

	#make a large DF	
	results = pandas.concat(subjectDFs)

	#which items errored out?
	errorDF = results.loc[results['error'].notnull()]
	errorDF = errorDF.loc[errorDF['error'] != 'Omitting the math phrase']
	print(errorDF)
	#write it out to a CSV if desired

	#order speakers by creakiness
 	resultsBySubject = results.groupby(['random_id'])
	rbs = resultsBySubject[['creakSum','sampleSum']].sum()
	rbs['creakProp'] = 	rbs['creakSum'] / rbs['sampleSum']
	rbs = rbs.sort(['creakProp'], ascending=[0])
	print(rbs)	
	rbs.to_csv('creakinessBySubject.csv')

	#order sentences by creakiness: this is a sanity check
	resultsByPhrase = results.groupby(['phraseIndex'])
	rbp = resultsByPhrase[['creakSum','sampleSum']].sum()
	rbp['creakProp'] = 	rbp['creakSum'] / rbp['sampleSum']
	rbp = rbp.sort(['creakProp'], ascending=[0])
	print(rbp)
	rbp.to_csv('creakinessBySentence.csv')