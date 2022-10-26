import json
import project.server.main.tasks as tasks

tasks.LOG_DATA = False

#%%
params = {'user_id': '8369aa11-d1e1-4b43-bdd9-ec9af726ef25', 
          'question': 'Who is the closest relative of the bonobo?', 
          'nOutputs': '100', 
          'min_year': '1860', 
          'max_year': '2020', 
          'pubmed': True}
params['keywords'] = ['bonobo']

results = tasks.process_question(params)
x = json.loads(results)

#for i in range(len(x['answer'])):
#    print(x['start_score'][str(i)], x['end_score'][str(i)], x['answer'][str(i)])
    

scores = []
for i in range(len(x['answer'])):
    scores.append((x['answer'][str(i)], 
                   float(x['start_score'][str(i)])+float(x['end_score'][str(i)])))

scores.sort(key=lambda tup: tup[1])
for i in scores:
    print(i[1], i[0])
    
# Chimpanzee average
total_sum = 0
n = 0
for i in scores:
    if i[0].startswith('chimpanzee'):
        total_sum += i[1]
        n += 1
print('Chimpanzee avg score:', total_sum / n)

# Pan paniscus average
total_sum = 0
n = 0
for i in scores:
    if i[0].startswith('pan paniscus'):
        total_sum += i[1]
        n += 1
print('Pan paniscus avg score:', total_sum / n)

# First valid answer, at a score of 4.24
# After that, 17 wrong answers
# Before that, 29 answers, 7 of them are correct or almost correct

#%%

params['question'] = 'Which gender is affected by breast cancer?'
params['keywords'] = ['breast', 'cancer']
results = tasks.process_question(params)
x = json.loads(results)

scores = []
for i in range(len(x['answer'])):
    scores.append((x['answer'][str(i)], 
                   float(x['start_score'][str(i)])+float(x['end_score'][str(i)])))

scores.sort(key=lambda tup: tup[1])
for i in scores:
    print(i[1], i[0])
    
# First correct answer at a score of 2.36
# After that, 2 incorrect results
# Before that, 11 answers; 7 of them correct
    
# %%

params['question'] = 'Which type of autophagy is triggered when the cell is under starvation?'
params['keywords'] = ['autophagy', 'starvation']
results = tasks.process_question(params)
x = json.loads(results)

scores = []
for i in range(len(x['answer'])):
    scores.append((x['answer'][str(i)], 
                   float(x['start_score'][str(i)])+float(x['end_score'][str(i)])))

scores.sort(key=lambda tup: tup[1])
for i in scores:
    print(i[1], i[0])
    
# I get many answers (45)
# The one I'm looking for has scores 2.35 and 7.23. Don't know about the others
# 22 answers below 2.35

# %%  

params['question'] = 'What could be an indicator of renal failure in nephropathia epidemica?'
params['keywords'] = ['nephropathia', 'epidemica', 'renal']
results = tasks.process_question(params)
x = json.loads(results)

scores = []
for i in range(len(x['answer'])):
    scores.append((x['answer'][str(i)], 
                   float(x['start_score'][str(i)])+float(x['end_score'][str(i)])))

scores.sort(key=lambda tup: tup[1])
for i in scores:
    print(i[1], i[0])
    
# Answer I'm looking for has scores of 14.15 and 13.18
# Many other answers: 'hemorrhagic fever' and 'puumala virus', 'hantavirus infection'
# The clearly wrong answers appear to be at below a score of 3    

#%%
    
params['question'] = 'Which compounds act on tau phosphatases?'
params['keywords'] = ['tau', 'phosphates']
results = tasks.process_question(params)
x = json.loads(results)

scores = []
for i in range(len(x['answer'])):
    scores.append((x['answer'][str(i)], 
                   float(x['start_score'][str(i)])+float(x['end_score'][str(i)])))

scores.sort(key=lambda tup: tup[1])
for i in scores:
    print(i[1], i[0])
    
# Very few answers (7; 100 abstracts selected)
# Not the answer I'm looking for; low scores in general
    
#%%

params['question'] = 'Which conditions are unique to children with photosensitivity disorders?'
params['keywords'] = ['children', 'photosensitivity disorders']
results = tasks.process_question(params)
x = json.loads(results)

scores = []
for i in range(len(x['answer'])):
    scores.append((x['answer'][str(i)], 
                   float(x['start_score'][str(i)])+float(x['end_score'][str(i)])))

scores.sort(key=lambda tup: tup[1])
for i in scores:
    print(i[1], i[0])