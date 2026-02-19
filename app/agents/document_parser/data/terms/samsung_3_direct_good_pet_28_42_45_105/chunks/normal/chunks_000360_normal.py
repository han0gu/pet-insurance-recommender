from langchain_core.documents import Document

chunk = Document(
    page_content=('. 건강식품, 보조식품, 보조치료제 및 Supplement 비용(치료를 목적으로 하는지 불 문합니다.) 9. 목욕 비용(약용 및 처방샴푸 '
 '값 포함) 및 벼룩, 진드기, 모낭충의 제거 비용'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 69},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000360',
              'chunk_char_len': 104,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
