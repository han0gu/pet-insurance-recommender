from langchain_core.documents import Document

chunk = Document(
    page_content=('| 항암약물치료 | 항암약물치료 | 연간6회한 | 30만원 |\n'
 '\uf000 제1항에서 정한 반려동물의료비보험금이란 「1. 반려동물의료비Ⅱ(강아지) 특별- 약관」에서 보상하는 의료비보험금 합계를 '
 '말합니다.\n'
 '- \uf000 제1항에서 정한 주요치료보험금은 제1항의 의료비에서 제2항의 반려동물의료비보\n'
 '- 험금 및 보험증권에 기재된 자기부담금을 차감한 금액에 보험증권에 기재된 보상\n'
 '- 비율을 곱한 금액이며, 보험증권에 기재된 치료구분별 각각의 지급한도 및 보상\n'
 '- 한도액에 따라 보상하여 드립니다. 단, "특정처치(이물제거)"로 인한 주요치료보'),
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
