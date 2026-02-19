from langchain_core.documents import Document

chunk = Document(
    page_content=('1) “장해”라 함은 상해 또는 질병에 대하여 치유된 후 신체에 남아있는 영구적인 정신 또는 육체의 훼손상태 및 기능상실 상태를 말한다. '
 '다만, 질병과 부상의 주 증상과 합병증상 및 이에 대한 치료를 받는 과정에서 일시적으로 나타나는 증상은 장해에 포함되지 않는다. 2) '
 '“영구적”이라 함은 원칙적으로 치유하는 때 장래 회 복할 가망이 없는 상태로서 정신적 또는 육체적 훼손 상태임이 의학적으로 인정되는 '
 '경우를 말한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 176},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000618',
              'chunk_char_len': 233,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
