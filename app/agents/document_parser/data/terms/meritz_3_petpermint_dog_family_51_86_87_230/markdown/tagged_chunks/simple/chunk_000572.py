from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5 | 피부질환 | AFB010 | 편평세포암종 |\n'
 '| 5 | 피부질환 | AFA011 | 항문주위선종 |\n'
 '| 5 | 피부질환 | AFB012 | 항문주위선암종 |\n'
 '| 5 | 피부질환 | AFA013 | 상세미상의 피부 신생물 (양성) |\n'
 '| 5 | 피부질환 | AFB013 | 상세미상의 피부 신생물 (악성) |\n'
 '| 5 | 피부질환 | AFC013 | 상세미상의 피부 신생물 (양성 또는 악성 이 불확실한) |\n'
 '| 5 | 피부질환 | AFA014 | 기타 피부 신생물 (양성) |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000572',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
