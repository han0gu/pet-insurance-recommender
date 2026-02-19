from langchain_core.documents import Document

chunk = Document(
    page_content=('5) 한다리의 3대관절중 관절 하나의 기능에 뚜렷한 장해 를 남긴 때 | 10\n'
 '6) 한다리의 3대관절중 관절 하나의 기능에 약간의 장해 를 남긴 때 | 5\n'
 '7) 한다리에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20\n'
 '8) 한다리에 가관절이 남아 약간의 장해를 남긴 때 | 10\n'
 '9) 한다리의 뼈에 기형을 남긴 때 | 5\n'
 '10) 한 다리가 5cm 이상 짧아지거나 길어진 때 | 3 0\n'
 '11) 한 다리가 3cm 이상 짧아지거나 길어진 때 | 15\n'
 '12) 한 다리가 1cm 이상 짧아지거나 길어진 때 | 5\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 218},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000774',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
