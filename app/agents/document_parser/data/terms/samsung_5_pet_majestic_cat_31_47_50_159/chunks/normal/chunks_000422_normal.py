from langchain_core.documents import Document

chunk = Document(
    page_content=('만15세 미만자의 경우 부득이 사고일부터 2년이 지난 후 에 성형수술이 가능하다는 진단을 받은 경우에는 그 진단으로 대체할 수 '
 '있습니다)을 받은 경우 아래에 정한 금액을 상해흉터복원(성형) 수술비로 보험수익자에게 지급합니 다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['skin', 'joint']},
 'indexing': {'chunk_id': 'chunk_000422',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
