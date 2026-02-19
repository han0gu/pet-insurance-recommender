from langchain_core.documents import Document

chunk = Document(
    page_content=('② 납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다음 날까지로 합니다. ③ 보험수익자와 계약자가 다른 '
 '경우 보험수익자에게도 제1항에 따른 내용을 알려 드립 니다. ④ 보험료 납입이 연체중이라도 계약의 해지 전에 발생한 보험금 지급사유에 '
 '대하여 회 사는 보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 38},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 168,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
