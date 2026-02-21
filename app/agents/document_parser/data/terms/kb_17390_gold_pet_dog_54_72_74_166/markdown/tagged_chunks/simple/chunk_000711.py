from langchain_core.documents import Document

chunk = Document(
    page_content=('- 장부를 열람할 수 있습니다.\n'
 '제22조(특별약관의 무효)# 계약을 맺을 때에보험의 목적에 이미 사고가 발생하였을 경우 이 특별약관은 무효로 하며 이미 납입한 이 '
 '특별약관의 보험료를 돌려 드립니다. 다만, 회사의 고의 또는 과실로 인하여 계약이 무효로 된 경우와 회사가 승낙 전에 무효임을 알았거나 '
 '알\n'
 '수 있었음에도 불구하고 보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다\n'
 '특# 음날부터 반환일까지의 기간에 대하여 회사는 이 특별약관의 보험계약대출이율을 연# 단위 복리로 계산한 금액을더하여 돌려 '
 '드립니다.별약관'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000711',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
