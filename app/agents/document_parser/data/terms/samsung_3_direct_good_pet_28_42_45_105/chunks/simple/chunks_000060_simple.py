from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약자 또 는 피보험자가 사실대로 알리는 것을 방해한 경우, '
 '계약자 또는 피보험자에게 사실 대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때. 다만, 보험설계 사 등의 행위가 없었다 '
 '하더라도 계약자 또는 피보험자가 사실대로 알리지 않거나 부실한 사항을 알렸다고 인정되는 경우에는 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000060',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
