from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 계약자,<br>피보험자 또는 보험수익자의 책임 있는 사유로 지급이 지연된 때에는 그 해당기<br>간에 대한 이자는 더하여 '
 '지급하지 않습니다. 다만, 회사는 계약자 등이 분쟁조<br>정을 신청했다는 사유만으로 이자지급을 거절하지 않습니다.<br>\uf000 '
 '계약자, 피보험자 또는 보험수익자는 제9조(알릴 의무 위반의 효과) 및 제2항의<br>보험금 지급사유조사와 관련하여 의료기관, '
 '국민건강보험공단, 경찰서 등 관공서<br>에 대한 회사의 서면에 의한 조사요청에 동의하여야 합니다'),
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
 'indexing': {'chunk_id': 'chunk_000809',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
