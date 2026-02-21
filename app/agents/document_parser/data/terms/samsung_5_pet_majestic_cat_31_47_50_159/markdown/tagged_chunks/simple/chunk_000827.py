from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나) 소장을 3/4 이상 잘라내었을 때 또는 잘라낸 소장의 길이가 3m 이상일 때\n'
 '- 다) 간장의 3/4 이상을 잘라내었을 때\n'
 '- 라) 양쪽 고환 또는 양쪽 난소를 모두 잃었을 때\n'
 '# 4) "흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의\n'
 '경우 중 하나에 해당하는 때를 말한다.- 가) 한쪽 폐 또는 한쪽 신장을 전부 잘라내었을 때\n'
 '- 나) 방광 기능상실로 영구적인 요도루, 방광루, 요관 장문합 상태\n'
 '- 다) 위, 췌장을 50% 이상 잘라내었을 때'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000827',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
