from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제6조(특별약관의 소멸)# 피보험자가 사망하였을경우에는 이 특별약관 계약도 소멸되며 회사는 "보험료 및해약환급금 산출방법서"에서 '
 '정하는 바에 따라 피보험자의 사망 당시 이 특별약관의\n'
 '계약자적립액 및 미경과보험료를 계약자에게 지급합니다.# 제7조(준용규정)# 이 특별약관에서정하지 않은 사항은 보통약관 제1절 일반조항을 '
 '따릅니다. 다만,이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계'),
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
 'indexing': {'chunk_id': 'chunk_000307',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
