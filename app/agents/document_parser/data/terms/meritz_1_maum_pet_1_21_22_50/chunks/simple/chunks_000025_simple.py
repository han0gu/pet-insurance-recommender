from langchain_core.documents import Document

chunk = Document(
    page_content=(': 파보 바이러스 감염, 디스템퍼 바이러스 감염, 파라 인플루엔자 감염, 전염성 간염, 아데노 바이러스 2형 감염, 광견병, 코로나 '
 '바이러스 감염, 렙토스피라 감염, 필라리 아(심장사상충) 감염, 인플루엔자 감염, 고양이범백혈구감소증, 고양이칼리시바이러 스감염증, '
 '고양이바이러스성비기관지염, 고양이백혈병바이러스감염증, 고양이헤르페 스바이러스감염증, 고양이클라미디아감염증 3'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000025',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
