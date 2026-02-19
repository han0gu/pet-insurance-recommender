from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 이 계약과 관련된 개인정보를 이 계약의 체결, 유지, 보험금 지급 등을 위하여 「개인정보 보호법」,「신용정보의 이용 및 보호에 '
 '관한 법률」등 관계 법령에 정한 경우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없이 수집, 이용, 조회 또 는 제공하지 '
 '않습니다. 다만, 회사는 이 계약의 체결, 유지, 보험금 지급 등을 위하여 위 관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 '
 '보험회사 및 보험관련 단체 등에 개인정보를 제공할 수 있습니다. ② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000171',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
