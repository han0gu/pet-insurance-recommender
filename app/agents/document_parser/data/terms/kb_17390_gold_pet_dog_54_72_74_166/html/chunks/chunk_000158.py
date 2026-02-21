from langchain_core.documents import Document

chunk = Document(
    page_content=(". 국가</p><br><p id='195' data-category='list' "
 "style='font-size:14px'>나.「한국은행법」에 따른 한국은행<br>다. 대통령령으로 정하는 금융회사<br>라. 「자본시장과 "
 '금융투자업에 관한 법률」 제9조제15항제3호에 따른 주권<br>상장법인(투자성 상품 중 대통령령으로 정하는 금융상품계약체결등<br>을 할 '
 '때에는 전문금융소비자와 같은 대우를 받겠다는 의사를 금융상<br>품판매업자등에게 서면으로 통지하는 경우만 해당한다)<br>마'),
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
