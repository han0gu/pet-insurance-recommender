from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 위, 대장(결장~직장) 또는 췌장의 전부를 잘라내었을 때 나) 소장을 3/4 이상 잘라내었을 때 또는 잘라낸 소장의 길이가 3m '
 '이상일 때 다) 간장의 3/4 이상을 잘라내었을 때 라) 양쪽 고환 또는 양쪽 난소를 모두 잃었을 때\n'
 '4) "흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의 경우 중 하나에 해당하는 때를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary', 'other']},
 'indexing': {'chunk_id': 'chunk_000965',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
