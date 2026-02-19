from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 귀의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 두 귀의 청력을 완전히 잃었을 때 | 80\n'
 '2) 한 귀의 청력을 완전히 잃고, 다른 귀의 청력에 심한 장해를 남긴 때 | 45\n'
 '3) 한 귀의 청력을 완전히 잃었을 때 | 25\n'
 '4) 한 귀의 청력에 심한 장해를 남긴 때 | 15\n'
 '5) 한 귀의 청력에 약간의 장해를 남긴 때 | 5\n'
 '6) 한 귀의 귓바퀴의 대부분이 결손된 때 | 1 0\n'
 '7) 평형기능에 장해를 남긴 때 | 10\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 204},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000712',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
