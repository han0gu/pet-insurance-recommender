from langchain_core.documents import Document

chunk = Document(
    page_content=('. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다. (다만, 질병의 발생일 로부터 과거 1년 이내의 예방접종 기록이 있는 '
 '경우에는 보상합니다.)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 69},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000362',
              'chunk_char_len': 85,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
