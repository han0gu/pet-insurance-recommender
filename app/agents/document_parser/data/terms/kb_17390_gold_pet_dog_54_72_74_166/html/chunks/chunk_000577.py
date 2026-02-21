from langchain_core.documents import Document

chunk = Document(
    page_content=(". 창상봉합술</h1><br><h1 id='73' style='font-size:20px'>치료비(안면/경부)(1일1회한, 연간3회한, "
 "급여)</h1><br><p id='74' data-category='paragraph' "
 "style='font-size:14px'>제1조(보험금의 지급사유)<br>\uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 "
 '상해의 직접결과로써, 그 치료<br>를 목적으로 "창상봉합술(급여)"를 받은 경우 1일 1회에 한하여 이 '
 "특별약관의</p><br><table id='75'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
