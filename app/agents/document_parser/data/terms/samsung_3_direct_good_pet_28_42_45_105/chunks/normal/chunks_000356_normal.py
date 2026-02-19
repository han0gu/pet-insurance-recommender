from langchain_core.documents import Document

chunk = Document(
    page_content=('. 대한민국 이외 지역에서 발생한 사고 및 손해 12. 수의사 자격이 없는 자의 치료행위로 인한 손해(수의사의 소견 및 처방에 의한 경 '
 '우도 동일) 및 그로 인하여 가중된 손해'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000356',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
