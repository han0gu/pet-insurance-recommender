from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약\n'
 '- 항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '반KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 105- 105 -려동물\uf000 회사는 제1항에 따라 계약자를 변경한 경우, '
 '변경된 계약자에게 보험증권 및 약관을 교부하고 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여# 드립니다.- '
 '제14조(보험나이 등)\n'
 '- \uf000 이 특별약관에서의 반려동물의 나이는 만나이를 기준으로 합니다.\n'
 '- \uf000 제1항의 만나이는 계약일 현재 반려동물의 실제 만나이를 기준으로 하며, 이후 매'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000506',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
