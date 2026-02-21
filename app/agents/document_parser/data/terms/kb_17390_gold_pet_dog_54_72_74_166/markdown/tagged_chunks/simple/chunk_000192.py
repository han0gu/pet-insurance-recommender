from langchain_core.documents import Document

chunk = Document(
    page_content=('중도인출 가능액 = 80만원(총 중도인출 가능액) - 10만원 = 70만원제37조(배당금의 지급)회사는 이 보험에 대하여 계약자에게 '
 '배당금을 지급하지 않습니다.보통약제7관 지정대리청구에 관한 사항관이 계약의 계약자, 피보험자 및 보험수익자가 모두 동일한 경우에 한하여 '
 '적용됩니다. 특별제38조(적용대상)# 제39조(지정대리청구인의 지정)\uf000 계약자는 계약체결할 때 또는 계약체결 이후 다음 각 호의 '
 '어느 하나에 해당하는\n'
 '자 중 2인이내에서 보험금의 대리청구인(이하, "지정대리청구인"이라 합니다)을'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000192',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
