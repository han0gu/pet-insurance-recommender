from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[계약자가 2명 이상인 경우] 계약자가 2명 이상인 경우 계약 전 알릴 의무, 보험료 납입의무 등 보험계약에 따른 계약자의 의무 二 '
 '···'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 31},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000041',
              'chunk_char_len': 84,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
