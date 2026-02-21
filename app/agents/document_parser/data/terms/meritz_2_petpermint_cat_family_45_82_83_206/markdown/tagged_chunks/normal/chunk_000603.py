from langchain_core.documents import Document

chunk = Document(
    page_content=('- 라) 양쪽 고환 또는 양쪽 난소를 모두 잃었을 때\n'
 '4) “흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를\n'
 '남긴 때”라 함은 아래의 경우 중 하나에 해당하는 때\n'
 '를 말한다.- 가) 한쪽 폐 또는 한쪽 신장을 전부 잘라내었을 때\n'
 '- 나) 방광 기능상실로 영구적인 요도루, 방광루, 요관\n'
 '- 장문합 상태\n'
 '- 다) 위, 췌장을 50% 이상 잘라내었을 때\n'
 '- 라) 대장절제, 항문 괄약근 등의 기능장해로 영구적으\n'
 '199로 장루, 인공항문을 설치한 경우(치료과정에서'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000603',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
