from langchain_core.documents import Document

chunk = Document(
    page_content=('청했다는 사유만으로 이자지급을 거절하지 않습니다.\n'
 '\uf000 계약자, 피보험자 또는 보험수익자는 제16조(알릴 의무 위반의 효과) 및 제2항의\n'
 '보험금 지급사유조사와 관련하여 의료기관, 국민건강보험공단, 경찰서 등 관공서\n'
 '에 대한 회사의 서면에 의한 조사요청에 동의하여야 합니다. 다만, 정당한 사유 없\n'
 '이 이에 동의하지 않을 경우 사실 확인이 끝날 때까지 회사는 보험금 지급지연에\n'
 '따른 이자를 지급하지 않습니다.- \n'
 '# 용 어 풀 이 정당한 사유| 의무의 이행을 | 당사자에게 사정이 |  | 기대하는 것이 무리라고 할 만한 있을 때(책 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
