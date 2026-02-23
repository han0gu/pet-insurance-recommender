from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>나.</p><br><p id='165' data-category='list' "
 "style='font-size:14px'>장해의 판정기준<br>1) ‘심장 기능을 잃었을 때’라 함은 심장 이식을 한 경우를 "
 '말한다.<br>2) ‘흉복부장기 또는 비뇨생식기 기능을 잃었을 때’라 함은 아래의 경우 중<br>하나에 해당하는 때를 '
 "말한다.</p><br><p id='166' data-category='paragraph' style='font-size:14px'>가) "
 '폐, 신장, 또는 간장의 장기이식을 한'),
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
