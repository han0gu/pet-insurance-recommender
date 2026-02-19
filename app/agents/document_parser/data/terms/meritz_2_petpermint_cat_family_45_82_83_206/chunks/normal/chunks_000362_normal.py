from langchain_core.documents import Document

chunk = Document(
    page_content=('인플루엔자 감염, 전염성 간염, 아데노 바이러스 2 형 감염, 광견병, 코로나 바이러스 감염, 렙토스피 라 감염, 필라리아(심장사상충) '
 '감염, 인플루엔자 감염, 고양이범백혈구감소증, 고양이칼리시바이러스 감염증, 고양이바이러스성비기관지염, 고양이백혈병 바이러스감염증, '
 '고양이헤르페스바이러스감염증, 고 양이클라미디아감염증'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 121},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000362',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.92}},
)
