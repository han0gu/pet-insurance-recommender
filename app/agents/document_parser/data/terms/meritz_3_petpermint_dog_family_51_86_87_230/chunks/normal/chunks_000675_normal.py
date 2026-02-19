from langchain_core.documents import Document

chunk = Document(
    page_content=('FAA006 | 비루관폐쇄\n'
 'FAA007 | 유루증\n'
 'FAA008 | 속눈썹의 질병 (첩모난생 / 첩모중생 / 이소성첩모)\n'
 'FAA009 | 안검내번·외번'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 195},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000675',
              'chunk_char_len': 83,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
