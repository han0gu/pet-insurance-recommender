from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항 제4호의 사고증명서는 수의사법 제2조(정의)에서 규정한 동물병원에서 수의사 가 발급한 것이어야 합니다.\n'
 '<관련법규>\n'
 '[수의사법 제2조(정의)]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 83},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000522',
              'chunk_char_len': 85,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
