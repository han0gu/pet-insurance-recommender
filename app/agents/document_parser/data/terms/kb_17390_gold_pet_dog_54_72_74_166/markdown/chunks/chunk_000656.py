from langchain_core.documents import Document

chunk = Document(
    page_content=('약성특KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 119- 119 -다.\n'
 '\uf000 반려동물(강아지) 일반조항에서 정하지 않은 사항은 보통약관 제1절 일반조항을따릅니다. 다만, 이 특별약관에서는 보통약관 '
 '제1절 일반조항 제9조(만기환급금의 지급), 제24조(계약의 소멸) 및 제36조(중도인출)는 제외합니다.| 5. '
 '반려동물배상책임(강아지)【갱신계약】 (【갱신계약】은 자동갱신으로 운영합니다) |\n'
 '| --- |'),
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
