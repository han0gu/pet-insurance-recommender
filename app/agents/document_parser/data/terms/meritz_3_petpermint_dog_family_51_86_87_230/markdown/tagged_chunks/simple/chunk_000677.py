from langchain_core.documents import Document

chunk = Document(
    page_content=('4) “흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를\n'
 '남긴 때”라 함은 아래의 경우 중 하나에 해당하는 때\n'
 '를 말한다.- 가) 한쪽 폐 또는 한쪽 신장을 전부 잘라내었을 때\n'
 '- 나) 방광 기능상실로 영구적인 요도루, 방광루, 요관\n'
 '- 장문합 상태\n'
 '- 다) 위, 췌장을 50% 이상 잘라내었을 때\n'
 '- 라) 대장절제, 항문 괄약근 등의 기능장해로 영구적으\n'
 '224로 장루, 인공항문을 설치한 경우(치료과정에서\n'
 '일시적으로 발생하는 경우는 제외)- 마) 심장기능 이상으로 인공심박동기를 영구적으로\n'
 '- 삽입한 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000677',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
