from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다) 위, 췌장을 50% 이상 잘라내었을 때\n'
 '- 라) 대장절제, 항문 괄약근 등의 기능장해로 영구적으로 장루, 인공항문을 설치\n'
 '- 한 경우(치료과정에서 일시적으로 발생하는 경우는 제외)\n'
 '- 마) 심장기능 이상으로 인공심박동기를 영구적으로 삽입한 경우\n'
 '- 바) 요도괄약근 등의 기능장해로 영구적으로 인공요도괄약근을 설치한 경우\n'
 '5) "흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때" 라 함은 아래의\n'
 '경우 중 하나에 해당하는 때를 말한다.- 가) 방광의 용량이 50cc 이하로 위축되었거나 요도협착, 배뇨기능 상실로 영구'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000828',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
