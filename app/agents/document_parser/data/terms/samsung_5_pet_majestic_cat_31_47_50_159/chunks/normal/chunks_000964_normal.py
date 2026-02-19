from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 폐, 신장, 또는 간장의 장기이식을 한 경우 나) 장기이식을 하지 않고서는 생명유지가 불가능하여 혈액투석, 복막투석 등 의료처치를 '
 '평생토록 받아야 할 때 다) 방광의 저장기능과 배뇨기능을 완전히 상실한 때\n'
 '3) "흉복부장기 또는 비뇨생식기 기능에 심한 장해를 남긴 때" 라 함은 아래의 경 우 중 하나에 해당하는 때를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['head', 'urinary', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000964',
              'chunk_char_len': 185,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
