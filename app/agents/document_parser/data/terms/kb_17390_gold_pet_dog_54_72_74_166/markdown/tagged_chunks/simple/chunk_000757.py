from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 이 특별약관의 부활(효력회복)청약을 받은 경우에는 보험계약의 부활(효력회\n'
 '복)을 승낙한 경우에 한하여 보통약관 제1절 일반조항 제29조(보험료의 납입을 연체\n'
 '하여 해지된 계약의 부활(효력회복)) 및 제30조(강제집행 등으로 인하여 해지된 계약\n'
 '의 특별부활(효력회복))에 따라 보험계약과 동시에 이 특별약관의 부활(효력회복)을\n'
 '취급합니다.제4조(준용규정)이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.2.선지급서비스# 제1조(적용대상)- \uf000 '
 '계약자와 동일한 피보험자에 대해서만 이 선지급서비스 특별약관(이하"특별약관"'),
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
 'indexing': {'chunk_id': 'chunk_000757',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
