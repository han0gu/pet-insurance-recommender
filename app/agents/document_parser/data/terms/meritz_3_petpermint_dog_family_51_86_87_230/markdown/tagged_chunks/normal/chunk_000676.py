from langchain_core.documents import Document

chunk = Document(
    page_content=('- 받아야 할 때\n'
 '- 다) 방광의 저장기능과 배뇨기능을 완전히 상실한 때\n'
 '3) “흉복부장기 또는 비뇨생식기 기능에 심한 장해를 남\n'
 '긴 때”라 함은 아래의 경우 중 하나에 해당하는 때를\n'
 '말한다.- 가) 위, 대장(결장∼직장) 또는 췌장의 전부를 잘라\n'
 '- 내었을 때\n'
 '- 나) 소장을 3/4이상 잘라내었을 때 또는 잘라낸 소장\n'
 '- 의 길이가 3m 이상일 때\n'
 '- 다) 간장의 3/4 이상을 잘라내었을 때\n'
 '- 라) 양쪽 고환 또는 양쪽 난소를 모두 잃었을 때\n'
 '4) “흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000676',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
