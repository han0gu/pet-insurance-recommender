from langchain_core.documents import Document

chunk = Document(
    page_content=('- 우 또는 반려동물(강아지) 일반조항 제22조(재가입) 제5항에 따라 이 특별약관\n'
 '- 계약이 연장된 경우에는 제6항 내지 제7항을 적용하지 않습니다.\n'
 '# 제2조(보험금 지급에 관한 세부규정)# \uf000 보험증권에 기재된 반려동물이동일한 날에 두 종류 이상의 "반려동물주요치료"를 받은 '
 '경우에는 하나의 주요치료보험금만 지급하며, 각 치료구분별 보상한도액\n'
 '중 최대 보상한도액에 해당하는 치료에 대하여 제1조(보험금의 지급사유) 제3항\n'
 '에 따라 보상하여 드립니다.|  | 동일한 날에 두 종류 이상의 "반려동물주요치료"를 받은 경우 |'),
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
