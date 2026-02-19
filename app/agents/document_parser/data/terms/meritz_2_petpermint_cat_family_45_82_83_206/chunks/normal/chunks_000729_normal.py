from langchain_core.documents import Document

chunk = Document(
    page_content=('6) 흉복부, 비뇨생식기계 장해는 질병 또는 외상의 직접 결과로 인한 장해를 말하며, 노화에 의한 기능장해 또는 질병이나 외상이 없는 '
 '상태에서 예방적으로 장 기를 절제, 적출한 경우는 장해로 보지 않는다. 7) 상기 흉복부 및 비뇨생식기계 장해항목에 명기되지 않은 기타 '
 '장해상태에 대해서는 “<붙임>일상생활 기본동작(ADLs) 제한 장해평가표”에 해당하는 장해 가 있을 때 ADLs 장해 지급률을 준용한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 200},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000729',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
