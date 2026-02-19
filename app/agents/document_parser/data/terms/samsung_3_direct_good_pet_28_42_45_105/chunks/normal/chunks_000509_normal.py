from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[핵연료물질]\n'
 '사용된 연료를 포함합니다. [핵연료물질에 의하여 오염된 물질] 원자핵 분열 생성물을 포함합니다.\n'
 '6. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해\n'
 '7. 최초계약의 보험계약일 이전에 이미 감염 또는 발병한 상해 및 질병'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 82},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000509',
              'chunk_char_len': 142,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
