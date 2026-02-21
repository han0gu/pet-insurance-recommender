from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이내에 정상으로 회복되는 발작을 말한다.\n'
 '229<붙임># 일상생활 기본동작(ADLs) 제한 장해평가표| 유형 | 제한정도에 따른 지급률 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000696',
              'chunk_char_len': 93,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
