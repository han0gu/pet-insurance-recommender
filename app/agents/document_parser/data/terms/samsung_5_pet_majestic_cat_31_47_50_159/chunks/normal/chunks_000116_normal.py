from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[계약자적립액] 장래의 보험금, 해약환급금 등을 지급하기 위하여 계약자가 납입한 보험료 중 일정액을 회사가 적 립해 둔 금액을 '
 '말합니다.\n'
 '제5관 보험료의 납입'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000116',
              'chunk_char_len': 95,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
