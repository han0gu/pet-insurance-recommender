from langchain_core.documents import Document

chunk = Document(
    page_content=('는 계약을 말합니다. 우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.# 제6조(특별약관의소멸)피보험자 또는 보험증권에 기재된 '
 '반려동물이 사망하였을 경우에는 이 특별약관 계\n'
 '약도 소멸되며 회사는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 피\n'
 '보험자 또는 보험증권에 기재된 반려동물의 사망 당시 이 특별약관의 계약자적립액'),
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
 'indexing': {'chunk_id': 'chunk_000728',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
