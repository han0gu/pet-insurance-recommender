from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 계약자 등이 분쟁조 정을 신청했다는 사유만으로 이자지급을 거절하지 않습니다. 4. 가산이율 적용시 보통약관 제1절 '
 '일반조항 제8조(보험금의 지급절차) 제2항 각 호의 어느 하나에 해당되는 사유로 지연된 경우에는 해당기간에 대하여 가산이율을 적용하지 '
 '않습니다. 5. 가산이율 적용시 금융위원회 또는 금융감독원이 정당한 사유로 인정하는 경 우에는 해당 기간에 대하여 가산이율을 적용하지 '
 '않습니다. 6'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000955',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
