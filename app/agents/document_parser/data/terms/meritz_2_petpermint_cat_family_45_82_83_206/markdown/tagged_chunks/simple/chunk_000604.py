from langchain_core.documents import Document

chunk = Document(
    page_content=('199로 장루, 인공항문을 설치한 경우(치료과정에서\n'
 '일시적으로 발생하는 경우는 제외)- 마) 심장기능 이상으로 인공심박동기를 영구적으로\n'
 '- 삽입한 경우\n'
 '- 바) 요도괄약근 등의 기능장해로 영구적으로 인공요\n'
 '- 도괄약근을 설치한 경우\n'
 '5) “흉복부장기 또는 비뇨생식기 기능에 약간의 장해를\n'
 '남긴 때”라 함은 아래의 경우 중 하나에 해당하는\n'
 '때를 말한다.- 가) 방광의 용량이 50cc 이하로 위축되었거나 요도협\n'
 '- 착, 배뇨기능 상실로 영구적인 간헐적 인공요도\n'
 '- 가 필요한 때\n'
 '- 나) 음경의 1/2 이상이 결손되었거나 질구 협착으로'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000604',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
