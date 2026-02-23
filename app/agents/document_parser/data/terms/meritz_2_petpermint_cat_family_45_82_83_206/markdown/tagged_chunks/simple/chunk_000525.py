from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 1) 두 귀의 청력을 완전히 잃었을 때 | 80 |\n'
 '| 2) 한 귀의 청력을 완전히 잃고, 다른 귀의 청력에 심한 장해를 남긴 때 | 45 |\n'
 '| 3) 한 귀의 청력을 완전히 잃었을 때 | 25 |\n'
 '| 4) 한 귀의 청력에 심한 장해를 남긴 때 | 15 |\n'
 '| 5) 한 귀의 청력에 약간의 장해를 남긴 때 | 5 |\n'
 '| 6) 한 귀의 귓바퀴의 대부분이 결손된 때 | 1 0 |\n'
 '| 7) 평형기능에 장해를 남긴 때 | 10 |\n'
 '# 나. 장해판정기준- 1) 청력장해는 순음청력검사 결과에 따라 데시벨(dB :'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000525',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
