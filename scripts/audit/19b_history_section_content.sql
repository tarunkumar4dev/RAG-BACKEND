-- Read-only: which NCERT chapter do the two Class 10 History section-title "chapters" actually contain?
select chapter, count(*) as rows,
  count(*) filter (where question_text ~* 'global|bretton|indentur|silk route|great depression|corn law|rinderpest|g-77|imf|world bank|columbus|el dorado') as global_world,
  count(*) filter (where question_text ~* 'industriali|proto-industr|factory|spinning|jenny|weaver|gomastha|jobber|mill') as industrialisation,
  count(*) filter (where question_text ~* 'print|press|book|gutenberg|newspaper|manuscript|censorship|vernacular|reading') as print_culture
from ncert_questions where class_grade = '10' and subject = 'History' group by 1 order by 1;
