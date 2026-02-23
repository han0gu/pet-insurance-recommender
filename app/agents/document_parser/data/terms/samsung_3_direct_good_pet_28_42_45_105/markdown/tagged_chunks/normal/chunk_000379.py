from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금 지급사유에 대해서는 보험금을 지급합니다.② 회사가 제1항에 따라 이 특별약관을 해지한 경우 회사는 그 취지를 계약자에게 통\n'
 '지하고 이 특별약관의 해약환급금을 지급합니다.제26조 (회사의 손해배상책임)- 75 -75 / 181- ① 회사는 계약과 관련하여 '
 '임직원, 보험설계사 및 대리점의 책임있는 사유로 계약자 및\n'
 '- 피보험자에게 발생된 손해에 대하여 관계 법령 등에 따라 손해배상의 책임을 집니다.\n'
 '- ② 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었는데도 소'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000379',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
