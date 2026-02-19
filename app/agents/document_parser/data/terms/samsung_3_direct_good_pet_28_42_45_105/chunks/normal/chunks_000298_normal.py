from langchain_core.documents import Document

chunk = Document(
    page_content=('제37조 (배당금의 지급)\n'
 '회사는 이 보험에 대하여 계약자에게 배당금을 지급하지 않습니다.\n'
 '제7관 분쟁의 조정 등\n'
 '제 38조 (분쟁의 조정)\n'
 '① 특별약관에 관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 금\n'
 '융감독원장에게 조정을 신청할 수 있으며, 분쟁조정 과정에서 계약자는 관계 법령이\n'
 '정하는 바에 따라 회사가 기록 및 유지·관리하는 자료의 열람(사본의 제공 또는 청취 를 포함한다)을 요구할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000298',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
