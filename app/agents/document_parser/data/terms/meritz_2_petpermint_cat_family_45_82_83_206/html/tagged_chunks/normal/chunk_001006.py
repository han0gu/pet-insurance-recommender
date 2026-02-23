from langchain_core.documents import Document

chunk = Document(
    page_content=('20° 이상인 경우를<br>말한다.<br>4) 갈비뼈(늑골)의 기형은 그 개수와 정도, 부위 등에 관<br>계없이 전체를 일괄하여 하나의 '
 '장해로 취급한다'),
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
 'indexing': {'chunk_id': 'chunk_001006',
              'chunk_char_len': 85,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
