from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사의 고의 또는 과실로 계약이 무효로 된 경우와 회사가 승낙<br>전에 무효임을 알았거나 알 수 있었음에도 보험료를 반환하지 '
 "않은 경우에는 보험료<br>를 납입한 날의 다음날부터 반환일까지의 기간에 대하여 회사는 이 계약의 보험계약</p><br><p id='3' "
 "data-category='list' style='font-size:14px'>대출이율을 연단위 복리로 계산한 금액을 더하여 돌려 "
 '드립니다.<br>1'),
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
 'indexing': {'chunk_id': 'chunk_000186',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
