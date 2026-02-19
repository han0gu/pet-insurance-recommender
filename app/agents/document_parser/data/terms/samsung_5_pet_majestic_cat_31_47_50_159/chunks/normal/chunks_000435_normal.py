from langchain_core.documents import Document

chunk = Document(
    page_content=('사고발생시점 만 15세 미만자의 경우 부득이 사고일 부터 2년이 지난 후에 성형수술이 가능하다는 진단을 받은 경우에는 그 진단으로 대 '
 '체할 수 있습니다)을 받은 경우 아래에 정한 금액을 안면부 상해흉터복원(성형) 수술 비로 보험수익자에게 지급합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 81},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000435',
              'chunk_char_len': 139,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
