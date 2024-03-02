# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="wzWM677Knout"
# # CS5004 ANALYTICS & SYSTEMS OF BIGT DATA PRACTICE – PROBLEM SET I
#
# Develop solutions for the following descriptive statistics scenarios using open-source development platforms:
#

# + id="hGwp9DMPk4SV"
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

rng = np.random.default_rng()


# + [markdown] id="cnlQv0lnnbTw"
# # 1.
#
# Given the following setup {Class, Tally score, Frequency}, develop an application that generates
# the table shown; (you can populate the relevant data; minimum data size :50 records). The table is only
# an illustration for a data of color scores, you are free to test the application over any data set with the
# application generating the tally and frequency scores.
#
# Score|Tally|Frequency
# -|-|-
# 1 = Brown|IIII\\ IIII\\ III|13
# 2 = Blue|IIII\\ III|8
# 3 = Green|IIII\\|5
# 4 = Other|II|2

# + id="2xnRokZko-r7"
def tally_str(n):
  return '||||\ ' * (n // 5) + '|' * (n % 5)

colors = ['Brown', 'Blue', 'Green', 'Other']


# + colab={"base_uri": "https://localhost:8080/", "height": 212} id="-U8_UOrOE_ZC" outputId="3f355b61-96a9-4b73-b5cb-dc9f85c1928b"
def q1():
  scores = rng.integers(low = 0, high = 4, size = 50)

  color_scores = [ colors[i] for i in scores ]
  print('color_scores:', color_scores)

  color_scores, counts = np.unique(color_scores, return_counts = True)
  freq_table = pd.DataFrame({ 'Score': color_scores, 'tally': [tally_str(i) for i in counts], 'Frequency': counts })
  freq_table = freq_table.sort_values(by = ['Frequency'], ascending = False)
  freq_table.index = range(1, len(freq_table) + 1)

  display(freq_table)

q1()


# + [markdown] id="uNqJLIbfrZr5"
# # 2.
#
# In a class of 18 students, assume marks distribution in an exam are as follows. Let the roll numbers
# start with CSE20D01 and all the odd roll numbers secure marks as follows: 25+((i+7)%10) and even
# roll numbers : 25+((i+8)%10). Develop an application that sets up the data and calculate the mean and
# median for the marks obtained using the platform support.

# + colab={"base_uri": "https://localhost:8080/"} id="ZEaGz3fJrkcJ" outputId="4d793bad-e040-4235-e546-170d0ac8b348"
def q2():
  total_students = 18
  roll_nos = np.arange(1, total_students + 1)

  marks_odd = 25 + (roll_nos[::2] + 7) % 10
  marks_even = 25 + (roll_nos[1::2] + 8) % 10

  marks = np.concatenate((marks_odd, marks_even))

  print(marks)
  print('Mean =', np.mean(marks))
  print('Median =', np.median(marks))

q2()


# + [markdown] id="_c3kcPYrvO4V"
# # 3.
#
# For a sample space of 20 elements, the values are fitted to the line Y=2X+3, X>5. Develop an
# application that sets up the data and computes the standard deviation of this sample space. (use random
# number generator supported in your development platform to generate values of X).

# + colab={"base_uri": "https://localhost:8080/"} id="CKJldwkZvS2I" outputId="ed8f117f-5917-4bca-909b-9d262d0d49d6"
def q3():
  x = rng.integers(low = 0, high = 10, size = 20)
  y = 2 * x[x > 5] + 3

  print('x =', x)
  print('y =', y)
  print('SD =', np.std(y))

q3()


# + [markdown] id="L2ON9DVUk-o0"
# # 4.
#
# For a given data of heights of a class, the heights of 15 students are recorded as 167.65, 167, 172,
# 175, 165, 167, 168, 167, 167.3, 170, 167.5, 170, 167, 169, and 172. Develop an application that
# computes; explore if there are any packages supported in your platform that depicts these measures /
# their calculations of central tendency in a visual form for ease of understanding.
#
# a. Mean height of the student
#
# b. Median and Mode of the sample space
#
# c. Standard deviation
#
# d. Measure of skewness. [(Mean-Mode)/standard deviation]

# + colab={"base_uri": "https://localhost:8080/", "height": 517} id="xNCCb6VblGco" outputId="2c4b93f4-73b6-43e0-b1ee-38f0acddb1ba"
def q4():
  h = [ 167.65, 167, 172, 175, 165, 167, 168, 167, 167.3, 170, 167.5, 170, 167, 169, 172 ]

  mean = np.mean(h)
  median = np.median(h)
  mode, count = stats.mode(h, keepdims = False)
  SD = np.std(h)
  skewness = (mean - mode) / SD

  print('a: Mean =', mean)
  print('b: Median =', median)
  print('Mode =', mode)
  print('c: SD =', SD)
  print('d: skewness =', skewness)

  plt.scatter(range(len(h)), h)
  plt.axhline(y = mean, label = 'mean')
  plt.axhline(y = median, label = 'median', color = 'green')
  plt.axhline(y = mode, label = 'mode', color = 'red')
  plt.legend()
  plt.show()

q4()


# + [markdown] id="eNcaCrXXwKvD"
# # 5.
#
# In Analytics and Systems of Bigdata course, for a class of 100 students, around 31 students secured
# ‘S’ grade, 29 secured ‘B’ grade, 25 ‘C’ grades, and rest of them secured ‘D’ grades. If the range of each
# grade is 15 marks. (S for 85 to 100 marks, A for 70 to 85 ...). Develop an application that represents
# the above data: using Pie and Bar graphs.

# + colab={"base_uri": "https://localhost:8080/", "height": 430} id="Khyrd4WiwNDi" outputId="3d13e13d-e169-45a6-e4d8-2bd70cbc3fad"
def q5():
  grades = [ 'S', 'A', 'B', 'C', 'D' ]
  all = [ 15, 0, 29, 25 ]
  all.append(100 - sum(all))

  plt.subplot(121)
  plt.pie(all, labels = grades)
  plt.legend()

  plt.subplot(122)
  plt.bar(grades, all)
  plt.show()

q5()


# + [markdown] id="CUpHamZC3UQt"
# # 6.
#
# On a given day (average basis), a student is observed to spend 33% of time in studying, 30% in
# sleeping, 18% in playing, 5% for hobby activities, and rest for spending with friends and family. Plot a
# pie chart showing his daily activities.

# + colab={"base_uri": "https://localhost:8080/", "height": 423} id="v5blS42N3V8X" outputId="1805625f-272e-4e5d-f85c-90aafcb9de2b"
def q6():
  labels = [
    'studying',
    'sleeping',
    'playing',
    'hobby activities',
    'spending time with friends and family'
  ]

  time = [33, 30, 18, 5]
  time.append(100 - sum(time))

  print(*zip(time, labels))

  plt.pie(time, labels = labels)
  plt.show()

q6()


# + [markdown] id="BUgwQrfR6jY0"
# # 7.
#
# Develop an application (absolute grader) that accepts marks scored by 20 students in ASBD course
# (as a split up of three: Mid Sem (30), End Sem (50) and Assignments(20).
#
# Compute the total and use it
# to grade the students following absolute grading: >=90 – S ; >=80 – A and so on till D.
#
# Compute the Class average for total marks in the course and 50% of class average would be fixed as the cut off for
# E. Generate a frequency table for the grades as well (Table displaying the grades and counts of them).

# + id="OeCFquJv4ncI"
def color_grades(grade):
  color = ''
  if grade == 'F':
    color = 'red'
  elif grade in ['S', 'A']:
    color = 'lawngreen'
  elif grade in ['B', 'C']:
    color = 'cyan'

  return ' background-color: %s' % color


def show_tables(marks):
  grades, counts = np.unique(marks['grades'], return_counts = True)
  freq_table = pd.DataFrame(counts, grades).transpose()
  display(freq_table)
  display(marks.style.applymap(color_grades, subset = [ 'grades' ]))


mid_sem = rng.integers(low = 0, high = 30, size = 20)
end_sem = rng.integers(low = 0, high = 50, size = 20)
assignments = rng.integers(low = 0, high = 20, size = 20)

marks = {
  'mid_sem': mid_sem,
  'end_sem': end_sem,
  'assignments': assignments,
}


# + colab={"base_uri": "https://localhost:8080/", "height": 758} id="dnEdbZCD6lYP" outputId="de53dbd5-25da-41ca-9edd-ccb953c3e701"
def absolute_grader(marks):
  marks['total'] = marks.sum(axis = 1)
  avg = marks['total'].mean()

  print('average:', avg)

  grades = []
  for mark in marks['total']:
    if mark >= 90:
      grades.append('S')
    elif mark >= 80:
      grades.append('A')
    elif mark >= 70:
      grades.append('B')
    elif mark >= 60:
      grades.append('C')
    elif mark >= 50:
      grades.append('D')
    elif mark >= avg / 2:
      grades.append('E')
    else:
      grades.append('F')

  marks['grades'] = grades
  show_tables(marks)


absolute_grader(pd.DataFrame(marks))


# + [markdown] id="7TopbCrPDS-b"
# # 8.
#
# Extend the application developed in (7) to support relative grading which uses the class average
# (mean) and standard deviation to compute the cutoffs for various grades as opposed to fixing them
# statically; you can refer the sample grader (excel sheet) attached to understand the formulas for fixing
# the cutoffs; the grader would involve, mean, standard deviation, max mark, passed students data mean,
# etc. Understand the excel grader thoroughly before you try mimicking such an application in your
# development platform.

# + colab={"base_uri": "https://localhost:8080/", "height": 897} id="7oKEvlnrDU98" outputId="c0013692-c1d8-49af-c204-0aaca575ca23"
def relative_grader(marks):
  marks['total'] = marks.sum(axis = 1)
  class_avg = marks['total'].mean()
  passing_minimum = class_avg / 2

  total = np.array(marks['total'])
  passing_students = total[total > passing_minimum]
  passing_students_mean = np.mean(passing_students)
  X = passing_students_mean - passing_minimum

  max_mark = np.max(marks['total'])
  S_cutoff = max_mark - 0.1 * (max_mark - passing_students_mean)
  Y = S_cutoff - passing_students_mean

  A_cutoff = passing_students_mean + Y * (5 / 8)
  B_cutoff = passing_students_mean + Y * (2 / 8)
  C_cutoff = passing_students_mean - X * (2 / 8)
  D_cutoff = passing_students_mean - X * (5 / 8)
  E_cutoff = passing_minimum

  grades = []
  for mark in marks['total']:
    if mark >= S_cutoff:
      grades.append('S')
    elif mark >= A_cutoff:
      grades.append('A')
    elif mark >= B_cutoff:
      grades.append('B')
    elif mark >= C_cutoff:
      grades.append('C')
    elif mark >= D_cutoff:
      grades.append('D')
    elif mark >= E_cutoff:
      grades.append('E')
    else:
      grades.append('F')

  marks['grades'] = grades

  print('class average =', class_avg)
  print('passing minimum =', passing_minimum)
  print('passing_students_mean =', passing_students_mean)
  print('S_cutoff =', S_cutoff)
  print('A_cutoff =', A_cutoff)
  print('B_cutoff =', B_cutoff)
  print('C_cutoff =', C_cutoff)
  print('D_cutoff =', D_cutoff)
  print('E_cutoff =', E_cutoff)

  show_tables(marks)


relative_grader(pd.DataFrame(marks))

# + [markdown] id="SrNhhp_q61xG"
# # Formulas Required for Relative Grading:
#
# - Passing Minimum: 50% of class average. (Minimum marks for passing)
# - X= Passing Students’ Mean- Passing Minimum.
# - S_cutoff = Max_Mark – 0.1 *(Max_Mark-Passing Students Mean)
# - Y= S_cutoff – Passing Students Mean
# - A_cutoff = Passing Students Mean + Y * (5/8)
# - B_cutoff = Passing Students Mean + Y * (2/8)
# - C_cutoff = Passing Students Mean - X * (2/8)
# - D_cutoff = Passing Students Mean - X * (5/8)
# - E_cutoff = Passing Minimum
#
# For the output data generated from Q7 and Q8, apply conditional formatting grade wise and highlight
# those who failed in red. (use minimum three-color codes)
