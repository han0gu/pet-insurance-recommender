from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 장해분류표에 해당되지 않는 후유장해는 피보험자의 직업, 연령, 신분 또는 성별 등에 관계없이 신체의 장해정도에 따라 장해분류표의 '
 '구분에 준하여 지급액을 결정합니다. 다만, 장해분류표의 각 장해분류별 최저 지급률 장해정도에 이르지 않는 후유장해에 대하여는 '
 '후유장해보험금을 지급하지 않습니다. ④ 보험수익자와 회사가 제3조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못할 때는 '
 '보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있 습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 29},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000013',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
