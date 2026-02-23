from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 회사는 피보험자가 그<br>사고에 관하여 가지는 항변으로써 피해자에게 대항할 수 있습니다.<br>\uf000 회사는 제1항의 '
 '청구를 받았을 때에는 지체없이 피보험자에게 통지하여야 하며,<br>회사의 요구가 있으면 피보험자 및 계약자는 필요한 서류・증거의 제출, '
 '증언 또<br>는 증인출석에 협조하여야 합니다.<br>\uf000 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 '
 '필요하다고<br>인정할 때에는 피보험자를 대신하여 회사의 비용으로 이를 해결할 수 있습니<br>다'),
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
 'indexing': {'chunk_id': 'chunk_001188',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
