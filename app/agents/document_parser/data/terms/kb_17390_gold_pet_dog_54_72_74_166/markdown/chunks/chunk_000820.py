from langchain_core.documents import Document

chunk = Document(
    page_content=('- 항 제9조(알릴 의무 위반의 효과) 등으로 보장을 제한하는 경우\n'
 '\uf000 제1항에 따라 보장을 제한하는 범위는 수의학적으로 인과관계가 있다고 입증된 경\n'
 '우 혹은 경험통계적으로 인과관계가 유의성있게 입증된 경우 등 해당 반려동물의\n'
 '과거 병력과 관련이 있는 특정 질병(【별표17】(반려동물(강아지) 특정 질병 분류\n'
 '표) 참조)으로 제한하여 적용하며, 그 판단기준은 회사에서 정한 계약사정기준(계\n'
 '약인수지침 등)을 따릅니다. 또한 회사는 이 특별약관이 부가된 경우 계약자에게\n'
 '보장제한부 인수 범위 및 사유를 설명하여 드립니다.'),
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
