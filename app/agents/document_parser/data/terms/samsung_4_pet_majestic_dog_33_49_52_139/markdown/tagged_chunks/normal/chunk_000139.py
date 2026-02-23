from langchain_core.documents import Document

chunk = Document(
    page_content=('- ④ 회사는 보험수익자에게 보험계약대출 사실을 통지할 수 있습니다.\n'
 '# 제38조 (배당금의 지급)회사는 이 보험에 대하여 계약자에게 배당금을 지급하지 않습니다.제7관 분쟁의 조정 등# 제 39조 (분쟁의 '
 '조정)- ① 계약에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 금융감\n'
 '- 독원장에게 조정을 신청할 수 있으며, 분쟁조정 과정에서 계약자는 관계 법령이 정하\n'
 '- 는 바에 따라 회사가 기록 및 유지·관리하는 자료의 열람(사본의 제공 또는 청취를 포\n'
 '- 함한다)을 요구할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000139',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
