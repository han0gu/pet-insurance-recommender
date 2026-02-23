from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그 밖에 대통령령으로 정하는 방식 관 ∙ 제33조(개인신용정보의 이용) 제2항 회사가 개인의 질병, 상해 또는 그 밖에 이와 유사한 '
 '정보를 수집ㆍ조사하거나 제3자에게 제공하는 경우 개인의 동의를 받아야 하며, 대통령령으로 정하는 목 적으로만 그 정보를 이용하여야 '
 "한다.</td></tr></tbody></table><h1 id='178' "
 "style='font-size:16px'>제50조(준거법)</h1><br><p id='179' "
 "data-category='paragraph' style='font-size:16px'>이 계약은"),
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
