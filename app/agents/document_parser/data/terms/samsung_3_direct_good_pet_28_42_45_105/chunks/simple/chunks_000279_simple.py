from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취지를 계약자에게 통지하고 제 35조(해약환급금) 제1항에 따른 해약환급금을 '
 '지급합니다. 다만, 제1항 제1호에서 보 험수익자가 보험금의 일부 보험수익자인 경우에는 지급하지 않은 보험금에 해당하는 해약환급금을 '
 '계약자에게 지급합니다.\n'
 '제34조 (회사의 파산선고와 해지)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 57},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000279',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
