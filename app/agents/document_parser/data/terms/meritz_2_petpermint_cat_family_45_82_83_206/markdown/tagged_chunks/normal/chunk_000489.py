from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5 | AFC008 | 흑색종 (양성 또는 악성이 불확실한) |  |\n'
 '| 5 | AFB009 | 피부 림프종 |  |\n'
 '| 5 | AFB010 | 편평세포암종 |  |\n'
 '| 5 | AFA011 | 항문주위선종 |  |\n'
 '| 5 | AFB012 | 항문주위선암종 |  |\n'
 '| 5 | AFA013 | 상세미상의 피부 신생물 (양성) |  |\n'
 '| 5 | AFB013 | 상세미상의 피부 신생물 (악성) |  |\n'
 '| 5 | AFC013 | 상세미상의 피부 신생물 (양성 또는 악성이 불확실한) |  |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000489',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
